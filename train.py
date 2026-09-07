import argparse
import csv
import json
import logging
import numpy as np
import os
import random
import torch
import torch.optim as optim

from models.swinunet import ChangeDetectionSwinUNet
from dataset.landslides import PSLandslideDataset
from dataset.lands2 import PSLandslideSentinel2Dataset
from dataset.contracts import validate_multimodal_batch
from dataset.multidata import MultiModalLandslideDataset
from dataset.sampler import BalancedPosNegSampler

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
PATIENCE = 20
MODEL_SELECTION_METRIC = "AUPRC"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TrainLogger")

# Runtime state is initialized only by main(), so importing validation helpers
# does not parse CLI arguments, scan datasets, create experiments or build a model.
EXPERIMENT_DIR = None
RUN_CONFIG = None
writer = None
device = None
START_EARLY_STOPPING_FROM_EPOCH = 0


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- DataLoader---------------------
def seed_worker(worker_id: int) -> None:
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# --------------------------- functions -------------------------------
def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train the multimodal landslide model")
    parser.add_argument("--description", required=True)
    parser.add_argument("--model", choices=["swinunet"], default="swinunet")
    parser.add_argument("--model-size", choices=["tiny", "small", "base"], default="small")
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches/"),
    )
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--train-events", nargs="+",
        default=["Philippines2019", "Michoacan2022", "EmiliaRomagna2023"],
    )
    parser.add_argument("--val-events", nargs="+", default=["Lombok2018"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    return parser.parse_args(argv)


def create_experiment_dir(args) -> Path:
    """Create a uniquely named experiment directory, or resume an existing one."""
    if args.resume:
        p = Path(args.resume)
        if not p.is_dir():
            raise FileNotFoundError(f"Cartella esperimento non trovata: {p}")
        return p

    base_name = f"{args.model}_{args.patch_size}"

    exp_root = Path("exp")
    exp_root.mkdir(exist_ok=True)

    experiment_dir = exp_root / base_name
    count = 2
    while experiment_dir.exists():
        experiment_dir = exp_root / f"{base_name}_{count}"
        count += 1

    experiment_dir.mkdir(parents=True, exist_ok=False)
    return experiment_dir

# -------------------- VALIDATION ---------------------
@torch.no_grad()
def validate(loader, model, criterion):
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
                batch, batch["s2_pre"].shape[1], "validation"
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
                    "aux", "s2_valid_pre", "s2_valid_post",
                ],
            )
            batch_contract_checked = True
        gt_mask = batch["mask"].to(device, non_blocking=True)

        logits = model(
            batch["s2_pre"].to(device, non_blocking=True),
            batch["s2_post"].to(device, non_blocking=True),
            batch["planet_pre"].to(device, non_blocking=True),
            batch["planet_post"].to(device, non_blocking=True),
            batch["aux"].to(device, non_blocking=True),
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
    if thresholds.numel() == 0:
        raise RuntimeError("PR curve has no threshold; validation labels are degenerate")
    # The final precision/recall point has no threshold and cannot select an
    # operating point. Each preceding point aligns with thresholds[index].
    best_idx = torch.argmax(f1[:-1]).item()
    best_thr = thresholds[best_idx].item()

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

def write_results(results):
    (EXPERIMENT_DIR / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )


def append_experiment(results):
    path = EXPERIMENT_DIR.parent / "experiments.csv"
    row = {"experiment": str(EXPERIMENT_DIR), **RUN_CONFIG, **results}
    existing = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8") as file:
            existing = list(csv.DictReader(file))
    fields = sorted({key for record in existing + [row] for key in record})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer_csv = csv.DictWriter(file, fieldnames=fields)
        writer_csv.writeheader()
        writer_csv.writerows(existing + [row])


# -------------------- TRAIN LOOP ---------------------
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs):
    best_model_score = float("-inf")
    best_threshold = 0.5
    best_epoch = None
    best_metrics = {}
    epochs_without_improvement = 0
    start_epoch = 0

    checkpoint_path = EXPERIMENT_DIR / "checkpoint_last.pth"
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as error:
            raise RuntimeError(
                "Checkpoint incompatibile con l'architettura multimodale AUX; "
                "avviare un nuovo training."
            ) from error
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_model_score = checkpoint["best_model_score"]
        best_threshold = checkpoint["best_threshold"]
        best_epoch = checkpoint.get("best_epoch")
        best_metrics = checkpoint.get("best_metrics", {})
        start_epoch = checkpoint["epoch"] + 1
        epochs_without_improvement = checkpoint.get("epochs_without_improvement", 0)
        logger.info(
            "Ripreso dal checkpoint: epoca %s, epoche senza miglioramento: %s",
            start_epoch,
            epochs_without_improvement,
        )

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss, total_samples = 0.0, 0
        batch_contract_checked = False

        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", ncols=100) as pbar:
            for batch in pbar:
                if not batch_contract_checked:
                    validate_multimodal_batch(
                        batch, batch["s2_pre"].shape[1], "training"
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
                            "aux", "s2_valid_pre", "s2_valid_post",
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
                    batch["aux"].to(device),
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
        is_best = model_score > best_model_score
        if is_best:
            best_model_score = model_score
            best_threshold = best_thr
            best_epoch = epoch + 1
            best_metrics = val_metrics.copy()
            epochs_without_improvement = 0
        elif epoch >= START_EARLY_STOPPING_FROM_EPOCH:
            epochs_without_improvement += 1

        logger.info(
            f"Epoch {epoch+1} | "
            f"Val loss: {val_metrics['val_loss']:.4f} | "
            f"AUPRC: {val_metrics['AUPRC']:.4f} | "
            f"F1: {val_metrics['F1']:.4f} | "
            f"IoU: {val_metrics['iou']:.4f}"
        )

        train_loss = total_loss / max(total_samples, 1)
        history_row = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics,
                       "threshold": best_thr, "best_score": best_model_score}
        history_path = EXPERIMENT_DIR / "history.csv"
        with history_path.open("a", newline="", encoding="utf-8") as file:
            writer_csv = csv.DictWriter(file, fieldnames=history_row.keys())
            if file.tell() == 0:
                writer_csv.writeheader()
            writer_csv.writerow(history_row)
        for name, value in history_row.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(name, value, epoch + 1)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_model_score': best_model_score,
            'best_threshold': best_threshold,
            'best_epoch': best_epoch,
            'best_metrics': best_metrics,
            'epochs_without_improvement': epochs_without_improvement,
            'metrics': val_metrics,
            'config': RUN_CONFIG,
        }
        torch.save(checkpoint, EXPERIMENT_DIR / "checkpoint_last.pth")

        if is_best:
            torch.save(checkpoint, EXPERIMENT_DIR / "best_model.pth")
            logger.info(f"Saved best model (AUPRC: {best_model_score:.4f})")

            # Salva PR curve CSV
            precision_np = pr_curve_data["precision"].numpy()
            recall_np    = pr_curve_data["recall"].numpy()
            thr_np       = pr_curve_data["thresholds"].numpy()
            with open(EXPERIMENT_DIR / "val_pr_curve.csv", "w", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(["threshold", "precision", "recall"])
                for t, p, r in zip(thr_np, precision_np[:-1], recall_np[:-1]):
                    writer_csv.writerow([t, p, r])

            # Salva PR curve PNG
            plot_pr_curve(
                pr_curve_data,
                best_idx,
                auprc=best_model_score,
                save_path=EXPERIMENT_DIR / "val_pr_curve.png"
            )

        if epoch >= START_EARLY_STOPPING_FROM_EPOCH and epochs_without_improvement >= PATIENCE:
            results = {
                "termination_reason": "early_stopping",
                "best_epoch": best_epoch,
                "best_score": best_model_score,
                "best_threshold": best_threshold,
                **{f"best_{key}": value for key, value in best_metrics.items()},
            }
            write_results(results)
            append_experiment(results)
            logger.info("Early stopping triggered")
            return

    results = {
        "termination_reason": "completed",
        "best_epoch": best_epoch,
        "best_score": best_model_score,
        "best_threshold": best_threshold,
        **{f"best_{key}": value for key, value in best_metrics.items()},
    }
    write_results(results)
    append_experiment(results)


def build_runtime(args):
    """Create training objects after CLI parsing, never at module import time."""
    global EXPERIMENT_DIR, RUN_CONFIG, START_EARLY_STOPPING_FROM_EPOCH
    global device, writer

    if args.epochs < 1 or args.warmup_epochs < 0:
        raise ValueError("--epochs deve essere >= 1 e --warmup-epochs >= 0")

    set_seed(SEED)
    generator = torch.Generator().manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    total_epochs = args.warmup_epochs + args.epochs
    START_EARLY_STOPPING_FROM_EPOCH = args.warmup_epochs + 1

    model = ChangeDetectionSwinUNet(
        model_size=args.model_size,
        img_size=args.patch_size,
    ).to(device)
    criterion = BCEWithLogitsLoss(
        pos_weight=torch.tensor([args.pos_weight], dtype=torch.float32, device=device)
    )
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    cosine = CosineAnnealingLR(optimizer, T_max=args.epochs)
    if args.warmup_epochs:
        warmup = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=args.warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine

    planet_train = PSLandslideDataset(
        args.dataset_root,
        args.train_events,
        patch_size=args.patch_size,
        apply_transform=False,
    )
    planet_val = PSLandslideDataset(
        args.dataset_root,
        args.val_events,
        patch_size=args.patch_size,
        apply_transform=False,
    )
    s2_train = PSLandslideSentinel2Dataset(
        args.dataset_root, args.train_events, apply_transform=False
    )
    s2_val = PSLandslideSentinel2Dataset(
        args.dataset_root, args.val_events, apply_transform=False
    )
    train_dataset = MultiModalLandslideDataset(
        planet_train, s2_train, apply_transform=True
    )
    val_dataset = MultiModalLandslideDataset(
        planet_val, s2_val, apply_transform=False
    )
    train_sampler = (
        BalancedPosNegSampler(train_dataset, args.patch_size)
        if args.balanced_sampling
        else None
    )
    logger.info(
        "Training sampler: %s; pos_weight=%s",
        "balanced" if train_sampler else "shuffle",
        args.pos_weight,
    )
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "generator": generator,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)

    EXPERIMENT_DIR = create_experiment_dir(args)
    logger.info("Experiment dir: %s", EXPERIMENT_DIR)
    writer = SummaryWriter(log_dir=EXPERIMENT_DIR / "logs")
    RUN_CONFIG = {
        "description": args.description,
        "model": args.model,
        "model_size": args.model_size,
        "patch_size": args.patch_size,
        "dataset_root": str(args.dataset_root),
        "train_events": args.train_events,
        "val_events": args.val_events,
        "epochs": total_epochs,
        "main_epochs": args.epochs,
        "warmup_epochs": args.warmup_epochs,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pos_weight": args.pos_weight,
        "balanced_sampling": args.balanced_sampling,
        "seed": SEED,
    }
    (EXPERIMENT_DIR / "config.json").write_text(
        json.dumps(RUN_CONFIG, indent=2), encoding="utf-8"
    )
    return model, train_loader, val_loader, criterion, optimizer, scheduler, total_epochs


def main(argv=None):
    try:
        runtime = build_runtime(parse_args(argv))
        train(*runtime)
    finally:
        if writer is not None:
            writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
