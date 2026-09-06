# make_patches_matched.py
# ------------------------------------------------------------
# Create PlanetScope patches and align ALL Sentinel-2 images
# spatially to Planet grid, preserving PRE/POST separation
# for downstream selection (Swin-UNet).
# ------------------------------------------------------------

import logging
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio import Affine

from skimage.util.shape import view_as_windows

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PATCH_SIZE = 64
OVERLAP = PATCH_SIZE // 2

PS_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches")
S2_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/Sentinel/images")
OUT_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/matchy")

EVENT_RANGES = {
    "Lombok2018": {
        "pre": (datetime(2018, 6, 1), datetime(2018, 8, 4)),
        "post": (datetime(2018, 8, 6), datetime(2018, 9, 30)),
    },
    "Philippines2019": {
        "pre": (datetime(2019, 6, 1), datetime(2019, 7, 24)),
        "post": (datetime(2019, 7, 26), datetime(2019, 9, 30)),
    },
    "Michoacan2022": {
        "pre": (datetime(2022, 8, 1), datetime(2022, 9, 18)),
        "post": (datetime(2022, 9, 20), datetime(2022, 10, 15)),
    },
    "EmiliaRomagna2023": {
        "pre": (datetime(2023, 4, 1), datetime(2023, 5, 15)),
        "post": (datetime(2023, 5, 17), datetime(2023, 6, 15)),
    },
}

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def get_crs_from_mgrs(tile_name: str):
    tile_name = tile_name.upper()
    zone = int(tile_name[1:3])
    band = tile_name[3]
    epsg = 32700 + zone if band < "N" else 32600 + zone
    return CRS.from_epsg(epsg)


def patchify(img: np.ndarray) -> np.ndarray:
    img = np.transpose(img, (1, 2, 0))
    patches = view_as_windows(
        img,
        (PATCH_SIZE, PATCH_SIZE, img.shape[-1]),
        step=(PATCH_SIZE - OVERLAP),
    )
    return np.transpose(patches.squeeze(2), (0, 1, 4, 2, 3))


def reproject_full(src_path: Path, ref_ds: rasterio.DatasetReader) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.zeros((src.count, ref_ds.height, ref_ds.width), dtype=np.float32)
        reproject(
            rasterio.band(src, list(range(1, src.count + 1))),
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_ds.transform,
            dst_crs=ref_ds.crs,
            resampling=Resampling.bilinear,
        )
    return dst


def write_patch(path: Path, data: np.ndarray, ref_ds: rasterio.DatasetReader, transform):
    meta = ref_ds.meta.copy()
    meta.update({
        "count": data.shape[0],
        "height": PATCH_SIZE,
        "width": PATCH_SIZE,
        "transform": transform,
        "dtype": data.dtype,
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)

# ------------------------------------------------------------
# MAIN PROCESSING
# ------------------------------------------------------------

def process_event(event: str):
    logging.info(f"Processing event {event}")

    ps_event = PS_BASE / event
    s2_event = S2_BASE / event / "combined"
    out_event = OUT_BASE / event

    for patch_dir in ps_event.iterdir():
        if not patch_dir.is_dir():
            continue

        planet_pre = patch_dir / "pre.tif"
        planet_post = patch_dir / "post.tif"
        dtm = patch_dir / "dtm.tif"
        mask = patch_dir / "y.tif"

        if not planet_pre.exists() or not planet_post.exists():
            continue

        with rasterio.open(planet_pre) as pre_ds, rasterio.open(planet_post) as post_ds:
            pre_patches = patchify(pre_ds.read())
            post_patches = patchify(post_ds.read())

            n_y, n_x = pre_patches.shape[:2]

            for iy in range(n_y):
                for ix in range(n_x):
                    pid = f"{patch_dir.name}_{iy}_{ix}"
                    out_patch = out_event / pid

                    offset_x = ix * (PATCH_SIZE - OVERLAP)
                    offset_y = iy * (PATCH_SIZE - OVERLAP)
                    patch_transform = pre_ds.transform * Affine.translation(offset_x, offset_y)

                    # --- Planet ---
                    write_patch(out_patch / "planet" / "pre.tif", pre_patches[iy, ix], pre_ds, patch_transform)
                    write_patch(out_patch / "planet" / "post.tif", post_patches[iy, ix], post_ds, patch_transform)

                    # --- Sentinel ---
                    for s2_tile in s2_event.iterdir():
                        for level in s2_tile.iterdir():
                            for date_dir in level.iterdir():
                                try:
                                    date = datetime.strptime(date_dir.name, "%Y%m%d")
                                except ValueError:
                                    continue

                                for phase, (d0, d1) in EVENT_RANGES[event].items():
                                    if not (d0 <= date <= d1):
                                        continue

                                    s2_path = date_dir / f"{date_dir.name}_10m.tif"
                                    if not s2_path.exists():
                                        continue

                                    s2_full = reproject_full(s2_path, pre_ds)
                                    s2_patch = s2_full[:, offset_y:offset_y+PATCH_SIZE, offset_x:offset_x+PATCH_SIZE]

                                    write_patch(
                                        out_patch / "s2" / phase / date_dir.name / "s2_10m.tif",
                                        s2_patch,
                                        pre_ds,
                                        patch_transform,
                                    )

                    # --- static layers ---
                    if dtm.exists():
                        shutil.copy2(dtm, out_patch / "dtm.tif")
                    if mask.exists():
                        shutil.copy2(mask, out_patch / "y.tif")


if __name__ == "__main__":
    for ev in EVENT_RANGES:
        process_event(ev)

    logging.info("All events processed")
