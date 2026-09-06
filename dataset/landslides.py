import json
import numpy as np
import random
import rasterio
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from shapely.geometry import box
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


PRE_FILENAME = "pre.tif"
POST_FILENAME = "post.tif"
DTM_FILENAME = "dem_wide.tif"
SLOPE_FILENAME = "slope_wide.tif"
ASPECT_FILENAME = "aspect_wide.tif"
MASK_FILENAME = "mask.tif"


class PSLandslideDataset(Dataset):
    def __init__(self, patches_dir, events, patch_size, apply_transform=False, use_post_only=False):
        self.patches_dir = Path(patches_dir)  # Ensure it's a Path object
        self.apply_transform = apply_transform
        self.patch_size = patch_size
        # keep a concrete list so we can iterate multiple times
        self.events = list(events)
        self.use_post_only = use_post_only

        cache_dir = Path("dataset/cache")
        cache_dir.mkdir(exist_ok=True, parents=True)

        events_tag = "_".join(sorted(events))
        self.cache_index_path = cache_dir / f"ps_index_{events_tag}.json"

        if self.cache_index_path.exists():
            # load and validate against requested events
            folders = self._load_folders_cache()
            cached_events = sorted({f["event"] for f in folders})
            requested_events = sorted(set(self.events))

            if cached_events == requested_events:
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

    def _read_folders(self, events):
        folders = []
        # Iterate over events with progress bar
        for event in tqdm(events, desc="Reading events"):
            event_dir = self.patches_dir / event

            if not event_dir.exists():
                continue  # Skip missing event directories

            patch_folders = sorted([p for p in event_dir.iterdir() if p.is_dir()])

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
                if all(p.exists() for p in [pre_path, post_path, dtm_path,
                                            slope_path, aspect_path, mask_path]):
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
            json.dump(serializable, f)

    def _load_folders_cache(self):
        """Load folder index from JSON cache."""
        with open(self.cache_index_path, "r") as f:
            data = json.load(f)
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

        # Stack auxiliary channels into a single tensor: [3,H,W]
        aux = torch.cat([dtm, slope, aspect], dim=0)

        if self.apply_transform:
            pre, post, aux, mask = self.transform(pre, post, aux, mask)

        return {
            "pre": pre,
            "post": post,
            "aux": aux,
            "mask": mask,
            "event": sample_paths["event"],
            "patch_id": sample_paths["patch_id"],
        }

    def transform(
        self,
        pre: torch.Tensor,
        post: torch.Tensor,
        aux: torch.Tensor,
        mask: torch.Tensor,
        crop_size=None
    ) -> tuple:

        # Horizontal flip
        if random.random() < 0.5:
            pre = TF.hflip(pre)
            post = TF.hflip(post)
            aux = TF.hflip(aux)
            mask = TF.hflip(mask)

        # Random rotation (0, 90, 180, 270 degrees)
        k = random.randint(0, 3)
        if k:
            dims = (1, 2)  # H, W
            pre = torch.rot90(pre, k, dims)
            post = torch.rot90(post, k, dims)
            aux = torch.rot90(aux, k, dims)
            mask = torch.rot90(mask, k, dims)

        # Optional: random crop
        if crop_size:
            i, j, h, w = TF.RandomCrop.get_params(pre, output_size=crop_size)
            pre = TF.crop(pre, i, j, h, w)
            post = TF.crop(post, i, j, h, w)
            aux = TF.crop(aux, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

        return pre, post, aux, mask

    def _read_tensor(self, file_path: Path) -> torch.Tensor:
        """Reads a raster file and converts it to a PyTorch tensor."""
        with rasterio.open(file_path) as src:
            data = src.read(out_dtype=np.float32)
            return torch.from_numpy(data)