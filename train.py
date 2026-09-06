import argparse
import csv
import logging
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.optim as optim

from models.swinunet import ChangeDetectionSwinUNet
from dataset.landslides import PSLandslideDataset
from dataset.lands2 import PSLandslideSentinel2Dataset
from dataset.contracts import validate_multimodal_batch
from dataset.multidata import MultiModalLandslideDataset
from dataset.sampler import BalancedPosNegSampler

from datetime import datetime
from pathlib import Path
from torch.nn import BCEWithLogitsLoss
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    BinaryPrecisionRecallCurve,
    BinaryAveragePrecision,
    BinaryAUROC
)
from torchmetrics.functional import jaccard_index
from tqdm import tqdm
from utils.plot import plot_pr_curve


# ------------------------ SEED -------------------------
SEED = 42


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# -------------------- DataLoader---------------------
def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


# -------------------- ARGS---------------------
parser = argparse.ArgumentParser()

parser.add_argument("--description", type=str, required=True)
parser.add_argument("--model", type=str, choices=["swinunet"], default="swinunet")
parser.add_argument("--model-size", type=str, choices=["tiny", "small", "base"], default="small")
parser.add_argument("--patch-size", type=int, default=128)
parser.add_argument("--dataset-root", type=str, default="/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches/")
parser.add_argument("--warmup-epochs", type=int, default=10)
parser.add_argument("--lr", type=float, default=1e-4)
parser.add_argument("--batch-size", type=int, default=2)
parser.add_argument("--train-events", type=str, nargs="+",
                    default=["Philippines2019", "Michoacan2022", "EmiliaRomagna2023"])
parser.add_argument("--val-events", type=str, nargs="+",
                    default=["Lombok2018"])
parser.add_argument("--resume", type=str, default=None)

args = parser.parse_args()

MODEL_SIZE = args.model_size
PATCH_SIZE = args.patch_size

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainLogger")

BATCH_SIZE = args.batch_size
MAIN_EPOCHS = 100
WARMUP_EPOCHS = args.warmup_epochs
TOTAL_EPOCHS = WARMUP_EPOCHS + MAIN_EPOCHS
PATIENCE = 20

SKIP_CHECKPOINT_BEFORE_EPOCH = WARMUP_EPOCHS
START_EARLY_STOPPING_FROM_EPOCH = WARMUP_EPOCHS + 1

LR = args.lr
POS_WEIGHT = 1.0

MODEL_SELECTION_METRIC = "AUPRC"
THRESHOLD_SELECTION_METRIC = "F1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------- EVENTS ---------------------
DATASET_DIR = Path(args.dataset_root)

TRAIN_EVENTS = args.train_events
VAL_EVENTS   = args.val_events


# -------------------- MODEL ---------------------
model = ChangeDetectionSwinUNet(
    model_size=MODEL_SIZE,
    img_size=PATCH_SIZE
).to(device)


# -------------------- LOSS / OPTIM ---------------------
pos_weight = torch.tensor([POS_WEIGHT], dtype=torch.float32).to(device)
criterion = BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=1e-4
)

warmup = LinearLR(optimizer, start_factor=0.1, total_iters=WARMUP_EPOCHS)
cosine = CosineAnnealingLR(optimizer, T_max=MAIN_EPOCHS)

scheduler = SequentialLR(
    optimizer,
    schedulers=[warmup, cosine] if WARMUP_EPOCHS > 0 else [cosine],
    milestones=[WARMUP_EPOCHS] if WARMUP_EPOCHS > 0 else []
)


# -------------------- DATASETS ---------------------
# Planet
planet_train = PSLandslideDataset(
    DATASET_DIR,
    TRAIN_EVENTS,
    patch_size=PATCH_SIZE,
    apply_transform=False
)

planet_val = PSLandslideDataset(
    DATASET_DIR,
    VAL_EVENTS,
    patch_size=PATCH_SIZE,
    apply_transform=False
)

# Sentinel-2
s2_train = PSLandslideSentinel2Dataset(
    DATASET_DIR,
    TRAIN_EVENTS,
    apply_transform=False
)

s2_val = PSLandslideSentinel2Dataset(
    DATASET_DIR,
    VAL_EVENTS,
    apply_transform=False
)

# Multimodal
train_dataset = MultiModalLandslideDataset(
    planet_train,
    s2_train,
    apply_transform=True
)

val_dataset = MultiModalLandslideDataset(
    planet_val,
    s2_val,
    apply_transform=False
)


# -------------------- SAMPLER & LOADER ---------------------
#train_sampler = BalancedPosNegSampler(train_dataset, PATCH_SIZE)
#sampler=train_sampler,
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=2,
    worker_init_fn=seed_worker,
    generator=g,
    pin_memory=False
)


# --------------------------- functions -------------------------------
def create_experiment_dir() -> Path:
    """Create a uniquely named experiment directory, or resume an existing one."""
    if args.resume:
        p = Path(args.resume)
        assert p.exists(), f"Cartella non trovata: {p}"
        return p

    base_name = f"{args.model}_{PATCH_SIZE}"

    exp_root = Path("exp")
    exp_root.mkdir(exist_ok=True)

    experiment_dir = exp_root / base_name
    count = 2
    while experiment_dir.exists():
        experiment_dir = exp_root / f"{base_name}_{count}"
        count += 1

    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir


def extract_criterion_params(crit: BCEWithLogitsLoss) -> dict:
    try:
        return {k: v for k, v in vars(crit).items() if not k.startswith("_")}
    except AttributeError:
        return {}


# -------------------- VALIDATION ---------------------
@torch.no_grad()
def validate(loader, model, criterion, th_metric="F1"):
    device = next(model.parameters()).device
    model.eval()

    pr_curve = BinaryPrecisionRecallCurve().to(device)
    auprc_m = BinaryAveragePrecision().to(device)
    auroc_m = BinaryAUROC().to(device)

    all_probs, all_masks = [], []
    running_loss, n_batches = 0.0, 0

    batch_contract_checked = False
    for batch in tqdm(loader, desc="Validating", ncols=100):
        if not batch_contract_checked:
            validate_multimodal_batch(
                batch, loader.dataset.s2_ds.n_temporal, "validation"
            )
            logger.info(
                "Validation DataLoader modalities: %s; model inputs: %s",
                {
                    key: tuple(value.shape)
                    for key, value in batch.items()
                    if isinstance(value, torch.Tensor)
                },
                [
                    "s2_pre", "s2_post", "planet_pre", "planet_post",
                    "s2_valid_pre", "s2_valid_post",
                ],
            )
            batch_contract_checked = True
        gt_mask = batch["mask"].to(device, non_blocking=True)

        logits = model(
            batch["s2_pre"].to(device, non_blocking=True),
            batch["s2_post"].to(device, non_blocking=True),
            batch["planet_pre"].to(device, non_blocking=True),
            batch["planet_post"].to(device, non_blocking=True),
            valid_t1=batch["s2_valid_pre"].to(device, non_blocking=True),
            valid_t2=batch["s2_valid_post"].to(device, non_blocking=True),
        )

        loss = criterion(logits, gt_mask.float())

        running_loss += loss.item()
        n_batches += 1

        probs = torch.sigmoid(logits)
        pr_curve.update(probs, gt_mask)
        auprc_m.update(probs, gt_mask)
        auroc_m.update(probs, gt_mask)

        all_probs.append(probs.cpu())
        all_masks.append(gt_mask.cpu())

    precision, recall, thresholds = pr_curve.compute()
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    best_idx = torch.argmax(f1).item()
    best_thr = thresholds[best_idx - 1].item() if best_idx > 0 else 0.0

    y_prob = torch.cat(all_probs).flatten()
    y_true = torch.cat(all_masks).flatten()
    y_pred = (y_prob >= best_thr).int()

    iou = jaccard_index(y_pred, y_true.int(), task="binary").item()

    pr_curve_data = {
        "precision": precision.cpu(),
        "recall": recall.cpu(),
        "thresholds": thresholds.cpu(),
    }

    return {
        "val_loss": running_loss / max(n_batches, 1),
        "AUPRC": auprc_m.compute().item(),
        "AUROC": auroc_m.compute().item(),
        "F1": f1[best_idx].item(),
        "precision": precision[best_idx].item(),
        "recall": recall[best_idx].item(),
        "iou": iou,
    }, best_thr, pr_curve_data, best_idx


EXPERIMENT_DIR = create_experiment_dir()
logger.info(f"Experiment dir: {EXPERIMENT_DIR}")

writer = SummaryWriter(log_dir=EXPERIMENT_DIR / "logs")


# -------------------- TRAIN LOOP ---------------------
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_model_score = 0
    best_threshold = 0.5
    epochs_without_improvement = 0
    start_epoch = 0

    checkpoint_path = EXPERIMENT_DIR / "checkpoint_last.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        best_model_score = checkpoint['best_model_score']
        best_threshold = checkpoint['best_threshold']
        start_epoch = checkpoint['epoch'] + 1
        epochs_without_improvement = checkpoint.get('epochs_without_improvement', 0)
        logger.info(f"Ripreso dal checkpoint: epoca {start_epoch}, epoche senza miglioramento: {epochs_without_improvement}")

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_samples = 0.0, 0
        batch_contract_checked = False

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
            for batch in pbar:
                if not batch_contract_checked:
                    validate_multimodal_batch(
                        batch, train_loader.dataset.s2_ds.n_temporal, "training"
                    )
                    logger.info(
                        "Training DataLoader modalities: %s; model inputs: %s",
                        {
                            key: tuple(value.shape)
                            for key, value in batch.items()
                            if isinstance(value, torch.Tensor)
                        },
                        [
                            "s2_pre", "s2_post", "planet_pre", "planet_post",
                            "s2_valid_pre", "s2_valid_post",
                        ],
                    )
                    batch_contract_checked = True
                gt_mask = batch["mask"].to(device)

                optimizer.zero_grad()
                logits = model(
                    batch["s2_pre"].to(device),
                    batch["s2_post"].to(device),
                    batch["planet_pre"].to(device),
                    batch["planet_post"].to(device),
                    valid_t1=batch["s2_valid_pre"].to(device),
                    valid_t2=batch["s2_valid_post"].to(device),
                )

                loss = criterion(logits, gt_mask.float())
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * gt_mask.size(0)
                total_samples += gt_mask.size(0)
                pbar.set_postfix(loss=total_loss / total_samples)

        scheduler.step()

        val_metrics, best_thr, pr_curve_data, best_idx = validate(
            val_loader, model, criterion
        )

        model_score = val_metrics[MODEL_SELECTION_METRIC]

        logger.info(
            f"Epoch {epoch+1} | "
            f"Val loss: {val_metrics['val_loss']:.4f} | "
            f"AUPRC: {val_metrics['AUPRC']:.4f} | "
            f"F1: {val_metrics['F1']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_model_score': best_model_score,
            'best_threshold': best_threshold,
            'epochs_without_improvement': epochs_without_improvement,
        }, EXPERIMENT_DIR / "checkpoint_last.pth")

        if epoch >= START_EARLY_STOPPING_FROM_EPOCH:
            if model_score > best_model_score:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= PATIENCE:
                    logger.info("Early stopping triggered")
                    break

        if model_score > best_model_score:
            best_model_score = model_score
            best_threshold = best_thr
            torch.save(model.state_dict(), EXPERIMENT_DIR / "best_model.pth")
            logger.info(f"Saved best model (AUPRC: {best_model_score:.4f})")

            # Salva PR curve CSV
            precision_np = pr_curve_data["precision"].numpy()
            recall_np    = pr_curve_data["recall"].numpy()
            thr_np       = pr_curve_data["thresholds"].numpy()
            thr_padded   = np.concatenate([[0.0], thr_np])
            with open(EXPERIMENT_DIR / "val_pr_curve.csv", "w", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(["threshold", "precision", "recall"])
                for t, p, r in zip(thr_padded, precision_np, recall_np):
                    writer_csv.writerow([t, p, r])

            # Salva PR curve PNG
            plot_pr_curve(
                recall_np,
                precision_np,
                auprc=best_model_score,
                save_path=EXPERIMENT_DIR / "val_pr_curve.png"
            )


if __name__ == "__main__":
    train(model, train_loader, val_loader, criterion, optimizer, scheduler, TOTAL_EPOCHS)
