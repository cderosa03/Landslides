import logging
import numpy as np
import os
import torch
import rasterio
import random
import torchvision.transforms.functional as TF
from datetime import datetime
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


# ── Parametri temporali ────────────────────────────────────────────────────
N_TEMPORAL = 6   # numero di immagini pre e post da caricare
MAX_EVENT_DISTANCE_DAYS = int(os.getenv("S2_MAX_EVENT_DISTANCE_DAYS", "90"))
REQUIRE_BOTH_TEMPORAL_SIDES = True

if MAX_EVENT_DISTANCE_DAYS < 0:
    raise ValueError("S2_MAX_EVENT_DISTANCE_DAYS must be non-negative")

logger = logging.getLogger(__name__)

# Data di taglio pre/post per ogni inventario (YYYYMMDD).
EVENT_CUTOFFS = {
    "Lombok2018":        "20180805",
    "Philippines2019":   "20191016",
    "Michoacan2022":     "20220919",
    "EmiliaRomagna2023": "20230516",
}


def select_temporal_dates(all_dates, cutoff, n_temporal):
    """Select real frames within the configured temporal distance from an event."""
    cutoff_date = datetime.strptime(cutoff, "%Y%m%d").date()
    eligible_dates = [
        date_str
        for date_str in all_dates
        if abs((datetime.strptime(date_str, "%Y%m%d").date() - cutoff_date).days)
        <= MAX_EVENT_DISTANCE_DAYS
    ]
    pre_dates = [date_str for date_str in eligible_dates if date_str < cutoff]
    post_dates = [date_str for date_str in eligible_dates if date_str >= cutoff]
    return pre_dates[-n_temporal:], post_dates[:n_temporal]


class PSLandslideSentinel2Dataset(Dataset):
    """
    Dataset Sentinel-2 con serie temporale.

    Per ogni patch restituisce:
      pre        : (N_TEMPORAL, 10, H, W)  — ultime N immagini prima dell'evento
      post       : (N_TEMPORAL, 10, H, W)  — prime N immagini dopo l'evento
      valid_pre  : (N_TEMPORAL,) bool      — True se frame reale, False se padding zero
      valid_post : (N_TEMPORAL,) bool

    Le bande sono 4×10m + 6×20m = 10 canali concatenati.
    La normalizzazione divide per 10000 (DN → riflettanza [0,1]).
    """

    def __init__(
        self,
        patches_root,
        events,
        apply_transform=False,
        normalize=True,
        n_temporal=N_TEMPORAL,
    ):
        self.root = Path(patches_root)
        self.events = list(events)
        self.apply_transform = apply_transform
        self.normalize = normalize
        self.n_temporal = n_temporal

        self.samples = self._collect_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                "Nessuna patch Sentinel-2 trovata. "
            )

    # ── Costruzione indice ────────────────────────────────────────────────
    def _collect_samples(self):
        samples = []

        for event in tqdm(self.events, desc="Lettura eventi S2"):
            cutoff = EVENT_CUTOFFS.get(event)
            if cutoff is None:
                print(f"[WARN] Nessun cutoff definito per '{event}', salto.")
                continue

            event_dir = self.root / event
            if not event_dir.exists():
                continue

            patch_dirs = sorted(
                [p for p in event_dir.iterdir() if p.is_dir() and p.name.isdigit()],
                key=lambda p: int(p.name),
            )

            for patch_dir in patch_dirs:
                s2_dir = patch_dir / "s2"
                if not s2_dir.exists():
                    continue

                # Raccogli tutte le date con entrambi i file
                all_dates = sorted([
                    d.name for d in s2_dir.iterdir()
                    if d.is_dir()
                    and (d / "s2_10m.tif").exists()
                    and (d / "s2_20m.tif").exists()
                    and (d / "s2_valid.tif").exists()
                ])

                if not all_dates:
                    continue

                # Dividi in pre e post rispetto al cutoff
                pre_dates, post_dates = select_temporal_dates(
                    all_dates, cutoff, self.n_temporal
                )

                # Tieni le N più recenti pre e le N più vecchie post

                # Il contratto multimodale richiede almeno una data reale
                # in entrambe le fasi. I frame oltre quelli disponibili
                # vengono invece gestiti con padding in _build_stack().
                if REQUIRE_BOTH_TEMPORAL_SIDES and (not pre_dates or not post_dates):
                    logger.info(
                        "Skipping %s/%s: incomplete temporal pair within %s days "
                        "(pre=%s, post=%s)",
                        event,
                        patch_dir.name,
                        MAX_EVENT_DISTANCE_DAYS,
                        pre_dates,
                        post_dates,
                    )
                    continue

                logger.info(
                    "Selected Sentinel dates for %s/%s: pre=%s post=%s",
                    event,
                    patch_dir.name,
                    pre_dates,
                    post_dates,
                )

                samples.append({
                    "event":      event,
                    "patch_id":   patch_dir.name,
                    "s2_dir":     s2_dir,
                    "pre_dates":  pre_dates,
                    "post_dates": post_dates,
                })

        return samples

    # ── Lettura ───────────────────────────────────────────────────────────
    def _read(self, path: Path) -> torch.Tensor:
        """Read a GeoTIFF while converting only its explicit NoData to zero."""
        with rasterio.open(path) as src:
            data = src.read(masked=True).astype(np.float32)
        return torch.from_numpy(np.ma.filled(data, fill_value=0.0))

    def _load_frame(self, s2_dir: Path, date: str) -> torch.Tensor:
        """Carica e concatena le bande 10m (4) e 20m (6) per una data → (10, H, W)."""
        t10 = self._read(s2_dir / date / "s2_10m.tif")  # (4, H, W)
        t20 = self._read(s2_dir / date / "s2_20m.tif")  # (6, H, W)
        return torch.cat([t10, t20], dim=0)              # (10, H, W)

    def _build_stack(self, s2_dir: Path, dates: list):
        """
        Costruisce uno stack (N_TEMPORAL, 10, H, W) dalle date disponibili.
        I frame mancanti vengono riempiti con zeri (padding).
        Restituisce (stack_tensor, valid_mask).
        """
        n = self.n_temporal
        frames = []
        valid  = []

        for date in dates:
            frames.append(self._load_frame(s2_dir, date))
            valid.append(True)

        # Padding con zeri se abbiamo meno di N_TEMPORAL date
        if not frames:
            raise RuntimeError(
                f"Nessun frame Sentinel-2 disponibile in {s2_dir}; "
                "il campione deve avere almeno una data reale."
            )
        ref_shape = frames[0].shape       # (10, H, W)

        while len(frames) < n:
            frames.append(torch.zeros(ref_shape, dtype=torch.float32))
            valid.append(False)

        stack      = torch.stack(frames, dim=0)                  # (N, 10, H, W)
        valid_mask = torch.tensor(valid, dtype=torch.bool)       # (N,)
        return stack, valid_mask

    # ── Dataset API ──────────────────────────────────────────────────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        pre,  valid_pre  = self._build_stack(s["s2_dir"], s["pre_dates"])
        post, valid_post = self._build_stack(s["s2_dir"], s["post_dates"])

        # Normalizzazione: DN Sentinel-2 → riflettanza [0, 1]
        if self.normalize:
            pre  = pre  / 10000.0
            post = post / 10000.0

        # Data augmentation 
        if self.apply_transform:
            pre, post = self._transform(pre, post)

        return {
            "pre":        pre,         # (N_TEMPORAL, 10, H, W)
            "post":       post,        # (N_TEMPORAL, 10, H, W)
            "valid_pre":  valid_pre,   # (N_TEMPORAL,) bool
            "valid_post": valid_post,  # (N_TEMPORAL,) bool
            "event":      s["event"],
            "patch_id":   s["patch_id"],
        }

    # ── Augmentation ─────────────────────────────────────────────────────
    def _transform(self, pre: torch.Tensor, post: torch.Tensor):
        """
        Flip orizzontale e rotazione casuale (0/90/180/270°) applicati
        coerentemente su tutti i frame della serie temporale.
        pre/post: (N, C, H, W)
        """
        if random.random() < 0.5:
            pre  = torch.flip(pre,  dims=[-1])
            post = torch.flip(post, dims=[-1])

        k = random.randint(0, 3)
        if k:
            pre  = torch.rot90(pre,  k, dims=[-2, -1])
            post = torch.rot90(post, k, dims=[-2, -1])

        return pre, post
