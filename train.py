import argparse
import csv
import faulthandler
from functools import partial
import json
import logging
import numpy as np
import os
import random
import time
from datetime import datetime, timezone
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
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.classification import (
    BinaryPrecisionRecallCurve,
    BinaryAveragePrecision,
    BinaryAUROC
)
from tqdm import tqdm
from utils.plot import plot_pr_curve
from utils.train_runtime import RunDiagnostics, atomic_torch_save, raster_environment


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
diagnostics = None
_worker_raster_env = None
_worker_trace_file = None


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- DataLoader---------------------
def seed_worker(worker_id: int, gdal_cache_mb=256, diagnostic_dir=None) -> None:
    global _worker_raster_env, _worker_trace_file
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
    torch.set_num_threads(1)
    # Spawned workers own their GDAL state and retain this bounded environment
    # for their lifetime. No CUDA calls are made in the worker.
    _worker_raster_env = raster_environment(gdal_cache_mb)
    _worker_raster_env.__enter__()
    if diagnostic_dir is not None:
        _worker_trace_file = (Path(diagnostic_dir) / f"worker_{os.getpid()}_stacks.log").open("a")
        faulthandler.enable(file=_worker_trace_file)
        # Periodic snapshots also expose a worker stuck inside a native read.
        faulthandler.dump_traceback_later(300, repeat=True, file=_worker_trace_file)


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
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gdal-cache-mb", type=int, default=256)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--log-dir", type=Path, default=Path("run_logs"))
    parser.add_argument(
        "--smoke-batches", type=int, default=0,
        help="Prova diagnostica: N batch training e fino a 4 validation, una sola epoca.",
    )
    parser.add_argument(
        "--loader-timeout",
        type=int,
        default=300,
        help=(
            "Secondi massimi di attesa per un batch con worker DataLoader; "
            "0 disabilita il timeout (default: 300)."
        ),
    )
    parser.add_argument(
        "--profile-batches",
        type=int,
        default=0,
        help="Misura i tempi per i primi N batch di ogni fase; 0 disabilita il profiling dettagliato.",
    )
    parser.add_argument(
        "--train-events", nargs="+",
        default=["Philippines2019", "Michoacan2022", "EmiliaRomagna2023"],
    )
    parser.add_argument("--val-events", nargs="+", default=["Lombok2018"])
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--pos-weight", type=float, default=1.0)
    parser.add_argument("--balanced-sampling", action="store_true")
    return parser.parse_args(argv)


def _cuda_synchronize() -> None:
    """Synchronize only when CUDA timings need to be measured."""
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _mark_progress():
    if diagnostics is not None:
        diagnostics.progress()


def _log_profile(scope, timing, profiled_batches, total_batches, total_seconds):
    """Log phase timings without adding synchronization to normal runs."""
    average_seconds = total_seconds / max(total_batches, 1)
    logger.info(
        "%s total: %.3fs; batches: %d; average batch: %.3fs",
        scope,
        total_seconds,
        total_batches,
        average_seconds,
    )
    if not profiled_batches:
        return
    labels = {
        "load": "load",
        "cpu_to_gpu": "cpu_to_gpu",
        "forward": "forward",
        "backward": "backward",
        "optimizer": "optimizer",
    }
    parts = [
        f"{labels[key]}={timing[key] * 1000 / profiled_batches:.3f}ms"
        for key in labels
        if key in timing
    ]
    logger.info(
        "%s profiled first %d batches: %s",
        scope,
        profiled_batches,
        "; ".join(parts),
    )


def create_experiment_dir(args) -> Path:
    """Create a uniquely named experiment directory, or resume an existing one."""
    if args.resume:
        p = Path(args.resume)
        if not p.is_dir():
            raise FileNotFoundError(f"Cartella esperimento non trovata: {p}")
        config_path = p / "config.json"
        if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")).get("diagnostic_only", False):
            raise ValueError("Un esperimento diagnostico non puo essere ripreso come training completo")
        return p

    base_name = f"{args.model}_{args.patch_size}"
    if args.smoke_batches:
        base_name += "_smoke"

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
def validate(loader, model, criterion, profile_batches=0):
    device = next(model.parameters()).device
    model.eval()

    # Fixed thresholds keep metric state bounded.  Without them TorchMetrics
    # stores every validation pixel until compute(), which can exhaust memory
    # on large validation sets.
    metric_thresholds = 512
    pr_curve = BinaryPrecisionRecallCurve(thresholds=metric_thresholds).to(device)
    auprc_m = BinaryAveragePrecision(thresholds=metric_thresholds).to(device)
    auroc_m = BinaryAUROC(thresholds=metric_thresholds).to(device)

    running_loss, n_batches = 0.0, 0
    phase_started = time.perf_counter()
    timing = {"load": 0.0, "cpu_to_gpu": 0.0, "forward": 0.0}
    profiled = 0

    batch_contract_checked = False
    loader_iter = iter(loader)
    with tqdm(total=len(loader), desc="Validating", ncols=100) as pbar:
        while True:
            _mark_progress()
            load_started = time.perf_counter()
            try:
                batch = next(loader_iter)
            except StopIteration:
                break
            load_seconds = time.perf_counter() - load_started
            profile_this = profile_batches > 0 and profiled < profile_batches
            if profile_this:
                _cuda_synchronize()

            transfer_started = time.perf_counter()
            gt_mask = batch["mask"].to(device, non_blocking=True)
            s2_pre = batch["s2_pre"].to(device, non_blocking=True)
            s2_post = batch["s2_post"].to(device, non_blocking=True)
            planet_pre = batch["planet_pre"].to(device, non_blocking=True)
            planet_post = batch["planet_post"].to(device, non_blocking=True)
            aux = batch["aux"].to(device, non_blocking=True)
            valid_t1 = batch["s2_valid_pre"].to(device, non_blocking=True)
            valid_t2 = batch["s2_valid_post"].to(device, non_blocking=True)
            if profile_this:
                _cuda_synchronize()
                timing["load"] += load_seconds
                timing["cpu_to_gpu"] += time.perf_counter() - transfer_started

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

            forward_started = time.perf_counter()
            logits = model(
                s2_pre,
                s2_post,
                planet_pre,
                planet_post,
                aux,
                valid_t1=valid_t1,
                valid_t2=valid_t2,
            )
            if profile_this:
                _cuda_synchronize()
                timing["forward"] += time.perf_counter() - forward_started

            loss = criterion(logits, gt_mask.float())

            running_loss += loss.item()
            n_batches += 1

            probs = torch.sigmoid(logits)
            pr_curve.update(probs, gt_mask)
            auprc_m.update(probs, gt_mask)
            auroc_m.update(probs, gt_mask)

            if profile_this:
                profiled += 1
            if profile_this or (n_batches % 50 == 0):
                logger.info("Validation batch %d/%d: load=%.3fs; total=%.3fs",
                            n_batches, len(loader), load_seconds,
                            time.perf_counter() - load_started)
            pbar.update(1)

    precision, recall, thresholds = pr_curve.compute()
    # Thresholds with no predicted positives can have undefined precision.
    # Treat undefined points as zero before argmax, so NaN cannot select the
    # operating threshold and propagate into F1/IoU or the checkpoint.
    precision = torch.nan_to_num(precision, nan=0.0)
    recall = torch.nan_to_num(recall, nan=0.0)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    if thresholds.numel() == 0:
        raise RuntimeError("PR curve has no threshold; validation labels are degenerate")
    # The final precision/recall point has no threshold and cannot select an
    # operating point. Each preceding point aligns with thresholds[index].
    best_idx = torch.argmax(f1[:-1]).item()
    best_thr = thresholds[best_idx].item()

    # For a binary confusion matrix, IoU = F1 / (2 - F1).  This avoids
    # retaining every prediction just to recompute IoU at the best threshold.
    best_f1 = f1[best_idx]
    iou = (best_f1 / (2.0 - best_f1 + 1e-9)).item()

    pr_curve_data = {
        "precision": precision.cpu(),
        "recall": recall.cpu(),
        "thresholds": thresholds.cpu(),
    }
    phase_seconds = time.perf_counter() - phase_started
    _log_profile(
        "Validation",
        timing,
        profiled,
        n_batches,
        phase_seconds,
    )

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
    if RUN_CONFIG.get("diagnostic_only", False):
        return
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
def train(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    epochs,
    profile_batches=0,
):
    best_model_score = float("-inf")
    best_threshold = 0.5
    best_epoch = None
    best_metrics = {}
    epochs_without_improvement = 0
    start_epoch = 0

    checkpoint_path = EXPERIMENT_DIR / "checkpoint_last.pth"
    training_started = time.perf_counter()
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if checkpoint.get("config", {}).get("diagnostic_only", False):
            raise ValueError("Un checkpoint diagnostico non puo essere ripreso come training completo")
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
        epoch_started = time.perf_counter()
        timing = {
            "load": 0.0,
            "cpu_to_gpu": 0.0,
            "forward": 0.0,
            "backward": 0.0,
            "optimizer": 0.0,
        }
        profiled = 0

        loader_iter = iter(train_loader)
        with tqdm(
            total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}", ncols=100
        ) as pbar:
            while True:
                _mark_progress()
                load_started = time.perf_counter()
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    break
                load_seconds = time.perf_counter() - load_started
                profile_this = profile_batches > 0 and profiled < profile_batches
                if profile_this:
                    _cuda_synchronize()

                transfer_started = time.perf_counter()
                gt_mask = batch["mask"].to(device, non_blocking=True)
                s2_pre = batch["s2_pre"].to(device, non_blocking=True)
                s2_post = batch["s2_post"].to(device, non_blocking=True)
                planet_pre = batch["planet_pre"].to(device, non_blocking=True)
                planet_post = batch["planet_post"].to(device, non_blocking=True)
                aux = batch["aux"].to(device, non_blocking=True)
                valid_t1 = batch["s2_valid_pre"].to(device, non_blocking=True)
                valid_t2 = batch["s2_valid_post"].to(device, non_blocking=True)
                if profile_this:
                    _cuda_synchronize()
                    timing["load"] += load_seconds
                    timing["cpu_to_gpu"] += time.perf_counter() - transfer_started

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

                optimizer.zero_grad(set_to_none=True)
                forward_started = time.perf_counter()
                logits = model(
                    s2_pre,
                    s2_post,
                    planet_pre,
                    planet_post,
                    aux,
                    valid_t1=valid_t1,
                    valid_t2=valid_t2,
                )
                if profile_this:
                    _cuda_synchronize()
                    timing["forward"] += time.perf_counter() - forward_started

                loss = criterion(logits, gt_mask.float())

                backward_started = time.perf_counter()
                loss.backward()
                if profile_this:
                    _cuda_synchronize()
                    timing["backward"] += time.perf_counter() - backward_started

                optimizer_started = time.perf_counter()
                optimizer.step()
                if profile_this:
                    _cuda_synchronize()
                    timing["optimizer"] += time.perf_counter() - optimizer_started

                total_loss += loss.item() * gt_mask.size(0)
                total_samples += gt_mask.size(0)
                pbar.set_postfix(loss=total_loss / total_samples)
                if profile_this:
                    profiled += 1
                if profile_this or ((pbar.n + 1) % 50 == 0):
                    logger.info(
                        "Training batch %d/%d: load=%.3fs; total=%.3fs",
                        pbar.n + 1,
                        len(train_loader),
                        load_seconds,
                        time.perf_counter() - load_started,
                    )
                pbar.update(1)

        epoch_seconds = time.perf_counter() - epoch_started
        _log_profile(
            f"Epoch {epoch + 1} training",
            timing,
            profiled,
            len(train_loader),
            epoch_seconds,
        )

        scheduler.step()

        val_metrics, best_thr, pr_curve_data, best_idx = validate(
            val_loader, model, criterion, profile_batches=profile_batches
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
        atomic_torch_save(checkpoint, EXPERIMENT_DIR / "checkpoint_last.pth")

        if is_best:
            atomic_torch_save(checkpoint, EXPERIMENT_DIR / "best_model.pth")
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
            total_seconds = time.perf_counter() - training_started
            logger.info(
                "Training total until early stopping: %.3fs (%.2fh)",
                total_seconds,
                total_seconds / 3600,
            )
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
    total_seconds = time.perf_counter() - training_started
    logger.info(
        "Training total: %.3fs (%.2fh)",
        total_seconds,
        total_seconds / 3600,
    )


def build_runtime(args):
    """Create training objects after CLI parsing, never at module import time."""
    global EXPERIMENT_DIR, RUN_CONFIG, START_EARLY_STOPPING_FROM_EPOCH
    global device, writer

    if args.epochs < 1 or args.warmup_epochs < 0:
        raise ValueError("--epochs deve essere >= 1 e --warmup-epochs >= 0")
    if args.profile_batches < 0:
        raise ValueError("--profile-batches deve essere >= 0")
    if args.num_workers < 0 or args.loader_timeout < 0:
        raise ValueError("--num-workers e --loader-timeout devono essere >= 0")
    if min(args.gdal_cache_mb, args.cpu_threads, args.batch_size, args.val_batch_size) < 1:
        raise ValueError("Cache, thread e batch size devono essere >= 1")
    if args.smoke_batches < 0:
        raise ValueError("--smoke-batches deve essere >= 0")
    if args.smoke_batches:
        if args.resume or args.balanced_sampling:
            raise ValueError("La prova diagnostica non supporta --resume o --balanced-sampling")
        args.epochs, args.warmup_epochs = 1, 0
        args.profile_batches = max(args.profile_batches, args.smoke_batches)
    torch.set_num_threads(args.cpu_threads)

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
    if args.smoke_batches:
        # Spread the test across the dataset using a separate deterministic RNG.
        probe_generator = torch.Generator().manual_seed(SEED)
        train_indices = torch.randperm(len(train_dataset), generator=probe_generator).tolist()
        val_indices = torch.randperm(len(val_dataset), generator=probe_generator).tolist()
        train_dataset = Subset(train_dataset, train_indices[:args.smoke_batches * args.batch_size])
        val_dataset = Subset(val_dataset, val_indices[:min(4, args.smoke_batches) * args.val_batch_size])
        logger.warning("DIAGNOSTIC ONLY: %d training samples, %d validation samples",
                       len(train_dataset), len(val_dataset))
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
        "num_workers": args.num_workers,
        "worker_init_fn": partial(seed_worker, gdal_cache_mb=args.gdal_cache_mb,
                                   diagnostic_dir=getattr(args, "diagnostic_dir", None)),
        "generator": generator,
        # Pinning is useful only for CUDA transfers and otherwise consumes
        # additional host memory.
        "pin_memory": args.pin_memory and device.type == "cuda",
    }
    if args.num_workers > 0:
        loader_options.update({
            # Do not keep the training worker pool alive while validation
            # starts a second pool.  This also releases GDAL/Rasterio caches
            # between phases and bounds host-memory use over long epochs.
            "persistent_workers": False,
            "prefetch_factor": 1,
            # GeoTIFF reads over NFS can occasionally exceed the usual batch time.
            "timeout": args.loader_timeout,
            # CUDA and GDAL have already been initialized in the parent.
            "multiprocessing_context": "spawn",
        })
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_options,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        **loader_options,
    )

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
        "val_batch_size": args.val_batch_size,
        "num_workers": args.num_workers,
        "loader_timeout": loader_options.get("timeout", 0),
        "pin_memory": loader_options["pin_memory"],
        "persistent_workers": loader_options.get("persistent_workers", False),
        "prefetch_factor": loader_options.get("prefetch_factor"),
        "multiprocessing_context": loader_options.get("multiprocessing_context"),
        "gdal_cache_mb": args.gdal_cache_mb,
        "cpu_threads": args.cpu_threads,
        "diagnostic_only": bool(args.smoke_batches),
        "smoke_batches": args.smoke_batches,
        "diagnostic_dir": str(getattr(args, "diagnostic_dir", "")),
        "profile_batches": args.profile_batches,
        "pos_weight": args.pos_weight,
        "balanced_sampling": args.balanced_sampling,
        "seed": SEED,
    }
    (EXPERIMENT_DIR / "config.json").write_text(
        json.dumps(RUN_CONFIG, indent=2), encoding="utf-8"
    )
    return (
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        total_epochs,
        args.profile_batches,
    )


def main(argv=None):
    global diagnostics
    args = parse_args(argv)
    # Spawned workers import NumPy/PyTorch afresh: bound native thread pools
    # before those imports, then set the main PyTorch pool in build_runtime().
    for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[variable] = "1"
    args.diagnostic_dir = args.log_dir / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ") + f"_{os.getpid()}"
    )
    args.diagnostic_dir.mkdir(parents=True, exist_ok=False)
    file_handler = logging.FileHandler(args.diagnostic_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    logger.info("Persistent diagnostics: %s", args.diagnostic_dir)
    logger.info("Arguments: %s", vars(args))
    try:
        with raster_environment(args.gdal_cache_mb), RunDiagnostics(args.diagnostic_dir) as run_diagnostics:
            diagnostics = run_diagnostics
            runtime = build_runtime(args)
            train(*runtime)
            if args.smoke_batches:
                logger.info("SMOKE TEST PASSED: training, validation and checkpoint completed")
    except BaseException:
        logger.exception("Training interrupted; diagnostics: %s", args.diagnostic_dir)
        raise
    finally:
        diagnostics = None
        if writer is not None:
            writer.close()
        logging.getLogger().removeHandler(file_handler)
        file_handler.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
