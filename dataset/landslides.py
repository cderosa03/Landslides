import json
from collections import Counter
from hashlib import sha256
import numpy as np
import rasterio
import torch

from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


PRE_FILENAME = "pre.tif"
POST_FILENAME = "post.tif"
DTM_FILENAME = "dem.tif"
SLOPE_FILENAME = "slope.tif"
ASPECT_FILENAME = "aspect.tif"
MASK_FILENAME = "mask.tif"

# AUX use rasters aligned to the Planet patch grid.
AUX_CHANNELS = ("dem_asinh", "slope_unit", "aspect_sin", "aspect_cos")
AUX_UNITS = ("normalized", "normalized", "unitless", "unitless")
AUX_NORMALIZATION = "asinh(dem_metres / 1000), slope_degrees / 90, sin/cos(aspect)"


class PSLandslideDataset(Dataset):
    def __init__(
        self,
        patches_dir,
        events,
        patch_size,
        apply_transform=False,
        use_post_only=False,
        patch_ids=None,
    ):
        if apply_transform:
            raise ValueError(
                "L'augmentation geometrica deve essere applicata da "
                "MultiModalLandslideDataset per restare sincronizzata."
            )
        self.patches_dir = Path(patches_dir)  # Ensure it's a Path object
        self.patch_size = patch_size
        # keep a concrete list so we can iterate multiple times
        self.events = list(events)
        self.patch_ids = None if patch_ids is None else {str(value) for value in patch_ids}
        self.use_post_only = use_post_only
        self.excluded = Counter()
        self.duplicate_ids = []

        cache_dir = Path("dataset/cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        self.cache_config = {
            "root": str(self.patches_dir.resolve()),
            "events": sorted(self.events),
            "patch_size": self.patch_size,
            "use_post_only": self.use_post_only,
            "patch_ids": None if self.patch_ids is None else sorted(self.patch_ids),
            "aux_channels": AUX_CHANNELS,
            "aux_grid": "planet_aligned",
        }
        cache_key = sha256(
            json.dumps(self.cache_config, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        self.cache_index_path = cache_dir / f"ps_index_{cache_key}.json"

        if self.cache_index_path.exists():
            # load and validate against requested events
            folders = self._load_folders_cache()
            if folders is not None:
                # cache is valid
                self.folders = folders
            else:
                # events changed → rebuild and overwrite cache
                self.folders = self._read_folders(self.events)
                self._save_folders_cache()
        else:
            # no cache yet → build and save
            self.folders = self._read_folders(self.events)
            self._save_folders_cache()

        if self.excluded:
            print(f"PlanetScope samples excluded: {dict(self.excluded)}")

    def _read_folders(self, events):
        folders = []
        seen = set()
        # Iterate over events with progress bar
        for event in tqdm(events, desc="Reading events"):
            event_dir = self.patches_dir / event

            if not event_dir.exists():
                continue  # Skip missing event directories

            patch_folders = sorted(
                p for p in event_dir.iterdir()
                if p.is_dir() and (self.patch_ids is None or p.name in self.patch_ids)
            )

            # Iterate over patch folders with nested progress bar
            for patch_folder in tqdm(
                patch_folders,
                desc=f"Reading patches for {event}",
                leave=False
            ):
                pre_path = patch_folder / PRE_FILENAME
                post_path = patch_folder / POST_FILENAME
                dtm_path = patch_folder / DTM_FILENAME
                slope_path = patch_folder / SLOPE_FILENAME
                aspect_path = patch_folder / ASPECT_FILENAME
                mask_path = patch_folder / MASK_FILENAME

                # Ensure all necessary files exist before adding
                paths = [pre_path, post_path, dtm_path, slope_path, aspect_path, mask_path]
                missing = [path.name for path in paths if not path.exists()]
                sample_id = (event, patch_folder.name)
                if missing:
                    self.excluded[f"missing:{','.join(missing)}"] += 1
                elif sample_id in seen:
                    self.duplicate_ids.append(sample_id)
                    self.excluded["duplicate_event_patch"] += 1
                else:
                    seen.add(sample_id)
                    folders.append({
                        "event": event,
                        "patch_id": patch_folder.name,
                        "pre": pre_path,
                        "post": post_path,
                        "dtm": dtm_path,
                        "slope": slope_path,
                        "aspect": aspect_path,
                        "mask": mask_path
                    })

        return folders

    def _save_folders_cache(self):
        """Save folder index to JSON cache."""
        self.cache_index_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = []
        for item in self.folders:
            serializable.append({
                "event": item["event"],
                "patch_id": item["patch_id"],
                "pre": str(item["pre"]),
                "post": str(item["post"]),
                "dtm": str(item["dtm"]),
                "slope": str(item["slope"]),
                "aspect": str(item["aspect"]),
                "mask": str(item["mask"]),
            })
        with open(self.cache_index_path, "w") as f:
            json.dump({"config": self.cache_config, "folders": serializable}, f)

    def _load_folders_cache(self):
        """Load folder index from JSON cache."""
        with open(self.cache_index_path, "r") as f:
            payload = json.load(f)
        if not isinstance(payload, dict) or payload.get("config") != self.cache_config:
            return None
        data = payload.get("folders", [])
        folders = []
        for item in data:
            # Extract patch_id from path if not in cache (backward compatibility)
            patch_id = item.get("patch_id")
            if patch_id is None:
                patch_id = Path(item["pre"]).parent.name
            folders.append({
                "event": item["event"],
                "patch_id": patch_id,
                "pre": Path(item["pre"]),
                "post": Path(item["post"]),
                "dtm": Path(item["dtm"]),
                "slope": Path(item["slope"]),
                "aspect": Path(item["aspect"]),
                "mask": Path(item["mask"]),
            })
        required = ("pre", "post", "dtm", "slope", "aspect", "mask")
        if any(not folder[name].exists() for folder in folders for name in required):
            return None
        return folders

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx: int) -> dict:
        sample_paths = self.folders[idx]
        pre = self._read_tensor(sample_paths["pre"])
        post = self._read_tensor(sample_paths["post"])
        dtm = self._read_tensor(sample_paths["dtm"])
        slope = self._read_tensor(sample_paths["slope"])
        aspect = self._read_tensor(sample_paths["aspect"])
        mask = self._read_tensor(sample_paths["mask"])

        # Normalize pre and post to [0, 1] range
        pre = pre.to(torch.float32) / 255.0
        post = post.to(torch.float32) / 255.0

        if self.use_post_only:
            pre = torch.zeros_like(pre)

        dtm = dtm.to(torch.float32)
        slope = slope.to(torch.float32)
        aspect = aspect.to(torch.float32)
        mask = (mask > 0).to(torch.uint8)

        # Keep terrain information stable without relying on split-specific
        # statistics: DEM retains absolute scale, slope is bounded, and aspect
        # is circular so 0Â° and 360Â° remain adjacent.
        dtm = torch.nan_to_num(dtm, nan=0.0, posinf=0.0, neginf=0.0)
        slope = torch.nan_to_num(slope, nan=0.0, posinf=90.0, neginf=0.0)
        aspect = torch.nan_to_num(aspect, nan=0.0, posinf=0.0, neginf=0.0)
        aspect_radians = torch.deg2rad(torch.remainder(aspect, 360.0))
        aux = torch.cat(
            [
                torch.asinh(dtm / 1000.0),
                torch.clamp(slope, 0.0, 90.0) / 90.0,
                torch.sin(aspect_radians),
                torch.cos(aspect_radians),
            ],
            dim=0,
        )

        return {
            "pre": pre,
            "post": post,
            "aux": aux,
            "mask": mask,
            "event": sample_paths["event"],
            "patch_id": sample_paths["patch_id"],
        }

    def _read_tensor(self, file_path: Path) -> torch.Tensor:
        """Reads a raster file and converts it to a PyTorch tensor."""
        with rasterio.open(file_path) as src:
            data = src.read(out_dtype=np.float32)
            return torch.from_numpy(data)
