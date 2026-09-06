# make_patches_matched.py
# ------------------------------------------------------------
# Match Sentinel-2 images to EXISTING Planet patches
# WITHOUT duplicating patches
# 1 patch = 1 folder
# ------------------------------------------------------------

import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime

import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from shapely.geometry import box
from rasterio.warp import transform_bounds

from tqdm import tqdm

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------

PS_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches")
S2_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/Sentinel/images")
OUT_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/matchati")

EVENTS = ["Lombok2018", "Philippines2019", "Michoacan2022", "EmiliaRomagna2023"]

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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------
# UTILITIES
# ------------------------------------------------------------

def get_crs_from_mgrs(tile_name: str) -> CRS:
    tile_name = tile_name.upper()
    zone = int(tile_name[1:3])
    band = tile_name[3]
    epsg = 32700 + zone if band < "N" else 32600 + zone
    return CRS.from_epsg(epsg)


def reproject_to_patch(src_path: Path, ref_ds: rasterio.DatasetReader) -> np.ndarray:
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


def write_patch(path: Path, data: np.ndarray, ref_ds: rasterio.DatasetReader):
    meta = ref_ds.meta.copy()
    meta.update({"count": data.shape[0], "dtype": data.dtype})
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(data)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def process_event(event: str):
    logging.info(f"Processing event: {event}")

    ps_event = PS_BASE / event
    s2_event = S2_BASE / event / "combined"
    out_event = OUT_BASE / event

    patch_dirs = sorted([d for d in ps_event.iterdir() if d.is_dir()])
    logging.info(f"Found {len(patch_dirs)} Planet patches")

    for patch_dir in tqdm(patch_dirs, desc=f"{event} patches"):
        pre_path = patch_dir / "pre.tif"
        post_path = patch_dir / "post.tif"

        if not pre_path.exists() or not post_path.exists():
            continue

        out_patch = out_event / patch_dir.name
        (out_patch / "planet").mkdir(parents=True, exist_ok=True)

        shutil.copy2(pre_path, out_patch / "planet" / "pre.tif")
        shutil.copy2(post_path, out_patch / "planet" / "post.tif")

        for aux in ["dtm.tif", "y.tif"]:
            if (patch_dir / aux).exists():
                shutil.copy2(patch_dir / aux, out_patch / aux)

        with rasterio.open(pre_path) as ref_ds:
            ref_bounds = box(*ref_ds.bounds)

            for tile_dir in s2_event.iterdir():
                tile_crs = get_crs_from_mgrs(tile_dir.name)

                for level in tile_dir.iterdir():
                    for date_dir in level.iterdir():
                        date = datetime.strptime(date_dir.name, "%Y%m%d")

                        for phase, (d0, d1) in EVENT_RANGES[event].items():
                            if not (d0 <= date <= d1):
                                continue

                            s2_path = date_dir / f"{date_dir.name}_10m.tif"
                            if not s2_path.exists():
                                continue

                            with rasterio.open(s2_path) as s2_ds:
                                s2_bounds = box(*transform_bounds(
                                    s2_ds.crs, ref_ds.crs, *s2_ds.bounds
                                ))

                                if not ref_bounds.intersects(s2_bounds):
                                    continue

                            s2_patch = reproject_to_patch(s2_path, ref_ds)
                            out_s2 = out_patch / "s2" / phase
                            write_patch(out_s2 / f"{date_dir.name}.tif", s2_patch, ref_ds)

    logging.info(f"Finished event {event}")


if __name__ == "__main__":
    for ev in EVENTS:
        process_event(ev)

    logging.info("All events processed successfully.")
