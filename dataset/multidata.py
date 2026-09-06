import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class MultiModalLandslideDataset(Dataset):
    """
    Dataset multimodale che allinea PlanetScope e Sentinel-2
    e restituisce input separati per ciascun encoder.

    Input PlanetScope : pre/post (C_p, H, W)
    Input Sentinel-2  : pre/post (N_TEMPORAL, 10, H, W) + valid_pre/post (N_TEMPORAL,)
    Ground truth      : mask (1, H, W) da PlanetScope
    """

    def __init__(self, planet_ds, s2_ds, apply_transform: bool = False):
        self.planet_ds = planet_ds
        self.s2_ds = s2_ds
        self.apply_transform = apply_transform

        self.aligned_indices = []

        # Costruisce mappa (event, patch_id) → indice PlanetScope
        # Legge direttamente dall'indice interno (niente I/O su disco)
        planet_map = {}
        for i, entry in enumerate(tqdm(self.planet_ds.folders,
                                       desc="Indicizzazione PlanetScope")):
            key = (entry["event"], entry["patch_id"])
            planet_map[key] = i

        # Allinea gli indici S2 con quelli PlanetScope
        for j, entry in enumerate(tqdm(self.s2_ds.samples,
                                       desc="Allineamento S2 ↔ PlanetScope")):
            key = (entry["event"], entry["patch_id"])
            if key in planet_map:
                self.aligned_indices.append({
                    "planet_idx": planet_map[key],
                    "s2_idx":     j,
                    "event":      entry["event"],
                    "patch_id":   entry["patch_id"],
                })

        print(f"MultiModal dataset pronto: {len(self.aligned_indices)} patch allineate")

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, idx):
        alignment = self.aligned_indices[idx]

        planet = self.planet_ds[alignment["planet_idx"]]
        s2     = self.s2_ds[alignment["s2_idx"]]

        # ── PlanetScope ───────────────────────────────────────────────────
        planet_pre  = planet["pre"]    # (C_p, H, W)
        planet_post = planet["post"]   # (C_p, H, W)
        mask        = planet["mask"]   # (1, H, W)  — ground truth

        # ── Sentinel-2 (serie temporale) ──────────────────────────────────
        s2_pre        = s2["pre"]         # (N_TEMPORAL, 10, H, W)
        s2_post       = s2["post"]        # (N_TEMPORAL, 10, H, W)
        s2_valid_pre  = s2["valid_pre"]   # (N_TEMPORAL,) bool
        s2_valid_post = s2["valid_post"]  # (N_TEMPORAL,) bool

        return {
            "planet_pre":    planet_pre,
            "planet_post":   planet_post,
            "s2_pre":        s2_pre,
            "s2_post":       s2_post,
            "s2_valid_pre":  s2_valid_pre,
            "s2_valid_post": s2_valid_post,
            "mask":          mask,
            "event":         alignment["event"],
            "patch_id":      alignment["patch_id"],
        }