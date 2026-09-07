import random

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset.contracts import validate_multimodal_sample


def sample_geometric_params(height, width, crop_size=None):
    """Sample one geometric transform shared by every spatial modality."""
    params = {
        "hflip": random.random() < 0.5,
        "rotation_k": random.randint(0, 3),
        "crop": None,
    }
    if crop_size is not None:
        crop_height, crop_width = crop_size
        rotated_height, rotated_width = (
            (width, height) if params["rotation_k"] % 2 else (height, width)
        )
        if not (0 < crop_height <= rotated_height and 0 < crop_width <= rotated_width):
            raise ValueError(
                f"crop_size={crop_size} non contenuto nella forma "
                f"spaziale {(rotated_height, rotated_width)}"
            )
        params["crop"] = (
            random.randint(0, rotated_height - crop_height),
            random.randint(0, rotated_width - crop_width),
            crop_height,
            crop_width,
        )
    return params


def apply_geometric_transform(tensor, params):
    """Apply one shared geometric transform to a tensor with H/W as last axes."""
    if params["hflip"]:
        tensor = torch.flip(tensor, dims=[-1])
    if params["rotation_k"]:
        tensor = torch.rot90(tensor, params["rotation_k"], dims=[-2, -1])
    if params["crop"] is not None:
        row, col, height, width = params["crop"]
        tensor = tensor[..., row:row + height, col:col + width]
    return tensor


class MultiModalLandslideDataset(Dataset):
    """Align PlanetScope and Sentinel-2 samples by event and patch identifier."""

    def __init__(self, planet_ds, s2_ds, apply_transform=False):
        self.planet_ds = planet_ds
        self.s2_ds = s2_ds
        self.apply_transform = apply_transform
        if apply_transform and (
            getattr(planet_ds, "apply_transform", False)
            or getattr(s2_ds, "apply_transform", False)
        ):
            raise ValueError(
                "Impostare apply_transform=False nei dataset Planet e Sentinel-2: "
                "l'augmentation è centralizzata nel dataset multimodale."
            )

        planet_indices = {}
        planet_duplicates = []
        for index, entry in enumerate(tqdm(planet_ds.folders, desc="Indicizzazione PlanetScope")):
            key = (entry["event"], entry["patch_id"])
            if key in planet_indices:
                planet_duplicates.append(key)
            else:
                planet_indices[key] = index
        s2_duplicates = []
        seen_s2 = set()
        for entry in s2_ds.samples:
            key = (entry["event"], entry["patch_id"])
            if key in seen_s2:
                s2_duplicates.append(key)
            seen_s2.add(key)
        if planet_duplicates or s2_duplicates:
            raise ValueError(
                f"Duplicati (event, patch_id): Planet={planet_duplicates}, S2={s2_duplicates}"
            )
        self.aligned_indices = [
            {
                "planet_idx": planet_indices[(entry["event"], entry["patch_id"])],
                "s2_idx": index,
                "event": entry["event"],
                "patch_id": entry["patch_id"],
            }
            for index, entry in enumerate(
                tqdm(s2_ds.samples, desc="Allineamento S2 ↔ PlanetScope")
            )
            if (entry["event"], entry["patch_id"]) in planet_indices
        ]
        self.alignment_stats = {
            "planet": len(planet_indices),
            "sentinel": len(s2_ds.samples),
            "aligned": len(self.aligned_indices),
            "planet_without_sentinel": len(set(planet_indices) - seen_s2),
            "sentinel_without_planet": len(seen_s2 - set(planet_indices)),
        }
        print(f"MultiModal dataset: {self.alignment_stats}")

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, index):
        alignment = self.aligned_indices[index]
        planet = self.planet_ds[alignment["planet_idx"]]
        sentinel = self.s2_ds[alignment["s2_idx"]]
        validate_multimodal_sample(
            planet, sentinel, alignment, self.s2_ds.n_temporal
        )

        if self.apply_transform:
            _, height, width = planet["pre"].shape
            params = sample_geometric_params(height, width)
            for name in ("pre", "post", "aux", "mask"):
                planet[name] = apply_geometric_transform(planet[name], params)
            for name in ("pre", "post"):
                sentinel[name] = apply_geometric_transform(sentinel[name], params)

        return {
            "planet_pre": planet["pre"],
            "planet_post": planet["post"],
            "aux": planet["aux"],
            "s2_pre": sentinel["pre"],
            "s2_post": sentinel["post"],
            "s2_valid_pre": sentinel["valid_pre"],
            "s2_valid_post": sentinel["valid_post"],
            "mask": planet["mask"],
            "event": alignment["event"],
            "patch_id": alignment["patch_id"],
        }
