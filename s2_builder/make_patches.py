import logging
import numpy as np
from pathlib import Path
from shapely.geometry import box
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.crs import CRS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


PS_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches/")
S2_BASE = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/Sentinel/images/")
EVENTS = ["Lombok2018", "Philippines2019", "Michoacan2022", "EmiliaRomagna2023"]


def get_crs_from_mgrs(tile_name: str):
    """Deduce EPSG from Sentinel-2 MGRS tile name (e.g., T50LLR → EPSG:32750/32650)"""
    try:
        tile_name = tile_name.upper()
        if not tile_name.startswith("T") or len(tile_name) < 4:
            return None
        zone = int(tile_name[1:3])
        band = tile_name[3]
        epsg = 32700 + zone if band < "N" else 32600 + zone
        crs = CRS.from_epsg(epsg)
        logging.info(f"Inferred CRS {crs} for tile {tile_name}")
        return crs
    except Exception as e:
        logging.warning(f"Cannot infer CRS from tile name {tile_name}: {e}")
        return None


def get_patch_bounds_utm(patch_ds: rasterio.io.DatasetReader, utm_epsg: int):
    """Returns patch bounding box in the given UTM projection"""
    patch_crs = patch_ds.crs or CRS.from_epsg(utm_epsg)
    bounds = patch_ds.bounds
    if patch_crs.to_epsg() != utm_epsg:
        bounds_utm = transform_bounds(patch_crs, CRS.from_epsg(utm_epsg), *bounds)
        return box(*bounds_utm)
    else:
        return box(bounds.left, bounds.bottom, bounds.right, bounds.top)


def find_overlapping_s2_tiles(event: str, patch_ds: rasterio.io.DatasetReader) -> list[Path]:
    """Find Sentinel-2 tiles overlapping a PlanetScope patch using UTM/MGRS"""
    s2_event_dir = S2_BASE / event / "combined"
    if not s2_event_dir.exists():
        logging.warning(f"No Sentinel-2 directory found for event {event}")
        return []

    overlapping_tiles = []

    for tile_dir in s2_event_dir.iterdir():
        if not tile_dir.is_dir():
            continue

        tile_crs = get_crs_from_mgrs(tile_dir.name)
        if tile_crs is None:
            continue

        patch_box = get_patch_bounds_utm(patch_ds, tile_crs.to_epsg())

        for level_dir in tile_dir.iterdir():  # MSIL1C / MSIL2A
            if not level_dir.is_dir():
                continue

            for date_dir in level_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                s2_file = date_dir / f"{date_dir.name}_10m.tif"
                if not s2_file.exists():
                    continue

                try:
                    with rasterio.open(s2_file) as ds:
                        ds_crs = ds.crs or tile_crs
                        s2_bounds = box(*transform_bounds(ds_crs, tile_crs, *ds.bounds))
                        if patch_box.intersects(s2_bounds):
                            overlapping_tiles.append(tile_dir)
                            logging.info(f"Found overlapping Sentinel-2 tile {tile_dir.name}")
                            break
                except Exception as e:
                    logging.warning(f"Error reading {s2_file}: {e}")
                    continue

    return overlapping_tiles


def get_all_s2_images_for_tile(tile_dir: Path) -> dict:
    """Search Sentinel-2 10m and 20m images in all levels (MSIL1C, MSIL2A)."""
    s2_images = {}

    for level_dir in tile_dir.iterdir():
        if not level_dir.is_dir():
            continue

        for date_dir in sorted(level_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            date_str = date_dir.name
            s2_10m = date_dir / f"{date_str}_10m.tif"
            s2_20m = date_dir / f"{date_str}_20m.tif"

            if s2_10m.exists() and s2_20m.exists():
                s2_images[date_str] = {"10m": s2_10m, "20m": s2_20m}

    return s2_images


def reproject_s2_to_patch(s2_path: Path, patch_ds: rasterio.io.DatasetReader) -> np.ndarray:
    """Reproject a Sentinel-2 image onto the PlanetScope patch grid (spatially aligned)."""
    try:
        with rasterio.open(s2_path) as src:
            if src.count == 0 or src.width == 0 or src.height == 0:
                raise ValueError("Source has no valid raster bands or dimensions")
            if src.transform is None:
                raise ValueError("Source transform is None")

            # CRS: usa quello del file o inferiscilo dal nome del tile
            tile_name = s2_path.parents[2].name
            src_crs = src.crs or get_crs_from_mgrs(tile_name)
            if src_crs is None:
                raise ValueError(f"Source CRS is None and cannot be inferred: {s2_path}")

            if patch_ds.crs is None:
                raise ValueError("Patch dataset has no CRS defined")

            dst_shape = (src.count, patch_ds.height, patch_ds.width)
            dst = np.zeros(dst_shape, dtype=np.float32)

            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=dst,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=patch_ds.transform,
                dst_crs=patch_ds.crs,
                resampling=Resampling.bilinear,
                num_threads=2,
                warp_mem_limit=512,
            )

        return dst

    except Exception as e:
        logging.warning(f"Skipping {s2_path}: {e}")
        return None


def write_patch(out_path: Path, data: np.ndarray, ref_ds: rasterio.io.DatasetReader):
    """Write a new GeoTIFF patch with metadata matching PlanetScope."""
    meta = ref_ds.meta.copy()
    meta.update({
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype,
        "transform": ref_ds.transform,
        "crs": ref_ds.crs,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)


def process_patch(patch_dir: Path, event: str):
    pre_path = patch_dir / "pre.tif"
    if not pre_path.exists():
        logging.warning(f"Missing pre.tif for patch {patch_dir.name}")
        return

    with rasterio.open(pre_path) as patch_ds:
        overlapping_tiles = find_overlapping_s2_tiles(event, patch_ds)

        if not overlapping_tiles:
            logging.warning(f"No overlapping Sentinel-2 tiles found for patch {patch_dir.name}")
            return

        s2_count = 0
        for tile_dir in overlapping_tiles:
            s2_images = get_all_s2_images_for_tile(tile_dir)
            for date_str, paths in s2_images.items():
                s2_10m_patch = reproject_s2_to_patch(paths["10m"], patch_ds)
                s2_20m_patch = reproject_s2_to_patch(paths["20m"], patch_ds)

                if s2_10m_patch is None or s2_20m_patch is None:
                    continue

                date_dir = patch_dir / "s2" / date_str
                write_patch(date_dir / "s2_10m.tif", s2_10m_patch, patch_ds)
                write_patch(date_dir / "s2_20m.tif", s2_20m_patch, patch_ds)
                s2_count += 1
                logging.info(f"Saved Sentinel-2 patch for {date_str} ({tile_dir.name})")

        if s2_count == 0:
            logging.warning(f"No Sentinel-2 data extracted for patch {patch_dir.name}")
        else:
            logging.info(f"Extracted {s2_count} Sentinel-2 images for patch {patch_dir.name}")


def process_event(event: str):
    ps_event_dir = PS_BASE / event
    if not ps_event_dir.exists():
        logging.warning(f"PlanetScope directory not found: {ps_event_dir}")
        return

    patch_dirs = [d for d in ps_event_dir.iterdir() if d.is_dir() and d.name.isdigit()]
    logging.info(f"Processing {len(patch_dirs)} patches for event {event}")

    for patch_dir in patch_dirs:
        process_patch(patch_dir, event)

    logging.info(f"Completed processing event {event}")


if __name__ == "__main__":
    for ev in EVENTS:
        logging.info("=" * 60)
        logging.info(f"Processing event: {ev}")
        logging.info("=" * 60)
        process_event(ev)

    logging.info("All events processed successfully.")
