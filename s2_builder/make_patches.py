from collections import defaultdict
import logging
import numpy as np
import os
from pathlib import Path
from shapely.geometry import box
import rasterio
from rasterio.warp import reproject, Resampling, transform_bounds
from rasterio.crs import CRS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PS_BASE = Path(os.getenv("PS_PATCHES_PATH", PROJECT_ROOT / "PlanetScope" / "patches"))
S2_BASE = Path(os.getenv("S2_IMAGES_PATH", PROJECT_ROOT / "Sentinel" / "images"))
S2_PRODUCT_LEVEL = "MSIL2A"
MIN_VALID_COVERAGE = float(os.getenv("S2_MIN_VALID_COVERAGE", "0.95"))
INVALID_SCL_CLASSES = frozenset(
    int(value)
    for value in os.getenv("S2_INVALID_SCL_CLASSES", "0,1,3,8,9,10").split(",")
)
NODATA_VALUE = -9999.0
EVENTS = os.getenv(
    "S2_EVENTS", "Lombok2018,Philippines2019,Michoacan2022,EmiliaRomagna2023"
).split(",")

if not 0.0 <= MIN_VALID_COVERAGE <= 1.0:
    raise ValueError("S2_MIN_VALID_COVERAGE must be between 0 and 1")


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

    for tile_dir in sorted(s2_event_dir.iterdir()):
        if not tile_dir.is_dir():
            continue

        tile_crs = get_crs_from_mgrs(tile_dir.name)
        if tile_crs is None:
            continue

        patch_box = get_patch_bounds_utm(patch_ds, tile_crs.to_epsg())

        level_dir = tile_dir / S2_PRODUCT_LEVEL
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
    """Search paired Sentinel-2 L2A 10m and 20m images for one tile."""
    s2_images = {}
    level_dir = tile_dir / S2_PRODUCT_LEVEL
    if not level_dir.is_dir():
        return s2_images

    for date_dir in sorted(level_dir.iterdir()):
        if not date_dir.is_dir():
            continue

        date_str = date_dir.name
        s2_10m = date_dir / f"{date_str}_10m.tif"
        s2_20m = date_dir / f"{date_str}_20m.tif"
        scl = date_dir / "sen2cor.tif"

        if s2_10m.exists() and s2_20m.exists() and scl.exists():
            s2_images[date_str] = {"10m": s2_10m, "20m": s2_20m, "scl": scl}
        elif s2_10m.exists() and s2_20m.exists():
            logging.warning("Skipping %s: missing Sen2Cor SCL mask", date_dir)

    return s2_images


def reproject_s2_to_patch(s2_path: Path, patch_ds: rasterio.io.DatasetReader):
    """Project one Sentinel-2 raster and its valid-data mask onto a Planet patch."""
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
            dst = np.full(dst_shape, NODATA_VALUE, dtype=np.float32)
            valid = np.zeros((patch_ds.height, patch_ds.width), dtype=np.uint8)

            reproject(
                source=rasterio.band(src, list(range(1, src.count + 1))),
                destination=dst,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=patch_ds.transform,
                dst_crs=patch_ds.crs,
                src_nodata=src.nodata,
                dst_nodata=NODATA_VALUE,
                init_dest_nodata=True,
                resampling=Resampling.bilinear,
                num_threads=2,
                warp_mem_limit=512,
            )
            reproject(
                source=src.dataset_mask(),
                destination=valid,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=patch_ds.transform,
                dst_crs=patch_ds.crs,
                resampling=Resampling.nearest,
                num_threads=2,
                warp_mem_limit=512,
            )

        return dst, valid.astype(bool)

    except Exception as e:
        logging.warning(f"Skipping {s2_path}: {e}")
        return None, None


def reproject_scl_to_patch(scl_path: Path, patch_ds: rasterio.io.DatasetReader):
    """Project the L2A Scene Classification Layer without interpolating classes."""
    try:
        with rasterio.open(scl_path) as src:
            src_crs = src.crs or get_crs_from_mgrs(scl_path.parents[2].name)
            if src_crs is None or patch_ds.crs is None:
                raise ValueError("SCL or Planet patch CRS is unavailable")

            scl = np.zeros((patch_ds.height, patch_ds.width), dtype=np.uint8)
            coverage = np.zeros_like(scl)
            reproject(
                source=src.read(1),
                destination=scl,
                src_transform=src.transform,
                src_crs=src_crs,
                src_nodata=src.nodata,
                dst_transform=patch_ds.transform,
                dst_crs=patch_ds.crs,
                dst_nodata=0,
                init_dest_nodata=True,
                resampling=Resampling.nearest,
                num_threads=2,
                warp_mem_limit=512,
            )
            reproject(
                source=src.dataset_mask(),
                destination=coverage,
                src_transform=src.transform,
                src_crs=src_crs,
                dst_transform=patch_ds.transform,
                dst_crs=patch_ds.crs,
                resampling=Resampling.nearest,
                num_threads=2,
                warp_mem_limit=512,
            )
        return scl, coverage.astype(bool)
    except Exception as error:
        logging.warning("Skipping SCL %s: %s", scl_path, error)
        return None, None


def mosaic_s2_images(image_paths, patch_ds):
    """Merge same-date tiles by taking the first valid pixel in sorted tile order."""
    mosaic_10m = mosaic_20m = valid = None

    for image_paths_for_tile in image_paths:
        data_10m, coverage_10m = reproject_s2_to_patch(
            image_paths_for_tile["10m"], patch_ds
        )
        data_20m, coverage_20m = reproject_s2_to_patch(
            image_paths_for_tile["20m"], patch_ds
        )
        scl, scl_coverage = reproject_scl_to_patch(image_paths_for_tile["scl"], patch_ds)
        if data_10m is None or data_20m is None or scl is None:
            continue

        if mosaic_10m is None:
            mosaic_10m = np.zeros_like(data_10m)
            mosaic_20m = np.zeros_like(data_20m)
            valid = np.zeros_like(coverage_10m, dtype=bool)

        tile_valid = (
            coverage_10m
            & coverage_20m
            & scl_coverage
            & ~np.isin(scl, tuple(INVALID_SCL_CLASSES))
        )
        take = tile_valid & ~valid
        mosaic_10m[:, take] = data_10m[:, take]
        mosaic_20m[:, take] = data_20m[:, take]
        valid |= tile_valid

    if mosaic_10m is None:
        return None, None, None

    mosaic_10m[:, ~valid] = NODATA_VALUE
    mosaic_20m[:, ~valid] = NODATA_VALUE
    return mosaic_10m, mosaic_20m, valid


def mosaic_metadata(date_str, image_paths):
    tiles = []
    products = []
    for paths in image_paths:
        with rasterio.open(paths["10m"]) as source:
            tags = source.tags()
        tiles.append(tags.get("s2_tile", paths["tile"]))
        products.append(tags.get("s2_product", paths["10m"].parent.name))

    return {
        "s2_level": S2_PRODUCT_LEVEL,
        "s2_acquisition_date": date_str,
        "s2_source_tiles": ",".join(tiles),
        "s2_source_products": ",".join(products),
    }


def write_patch(
    out_path: Path,
    data: np.ndarray,
    ref_ds: rasterio.io.DatasetReader,
    metadata: dict,
):
    """Write a new GeoTIFF patch with metadata matching PlanetScope."""
    meta = ref_ds.meta.copy()
    meta.update({
        "count": data.shape[0],
        "height": data.shape[1],
        "width": data.shape[2],
        "dtype": data.dtype,
        "transform": ref_ds.transform,
        "crs": ref_ds.crs,
        "nodata": NODATA_VALUE,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(data)
        dst.update_tags(**metadata)


def write_coverage(
    out_path: Path,
    valid: np.ndarray,
    ref_ds: rasterio.io.DatasetReader,
    metadata: dict,
):
    meta = ref_ds.meta.copy()
    meta.update({
        "count": 1,
        "height": valid.shape[0],
        "width": valid.shape[1],
        "dtype": "uint8",
        "transform": ref_ds.transform,
        "crs": ref_ds.crs,
        "nodata": 0,
    })
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(valid.astype(np.uint8), 1)
        dst.update_tags(**metadata)


def group_images_by_date(tile_dirs):
    """Group paired L2A rasters by date, preserving deterministic tile order."""
    images_by_date = defaultdict(list)
    for tile_dir in sorted(tile_dirs):
        for date_str, paths in get_all_s2_images_for_tile(tile_dir).items():
            images_by_date[date_str].append({"tile": tile_dir.name, **paths})
    return dict(sorted(images_by_date.items()))


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
        for date_str, image_paths in group_images_by_date(overlapping_tiles).items():
            s2_10m_patch, s2_20m_patch, valid = mosaic_s2_images(
                image_paths, patch_ds
            )
            if s2_10m_patch is None:
                continue

            coverage = float(valid.mean())
            invalid_fraction = 1.0 - coverage
            if coverage < MIN_VALID_COVERAGE:
                logging.warning(
                    "Skipping %s/%s on %s: valid %.2f%%, invalid %.2f%%",
                    event,
                    patch_dir.name,
                    date_str,
                    coverage * 100,
                    invalid_fraction * 100,
                )
                continue

            date_dir = patch_dir / "s2" / date_str
            date_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                **mosaic_metadata(date_str, image_paths),
                "s2_valid_coverage": f"{coverage:.6f}",
                "s2_invalid_fraction": f"{invalid_fraction:.6f}",
                "s2_invalid_scl_classes": ",".join(map(str, sorted(INVALID_SCL_CLASSES))),
            }
            write_patch(date_dir / "s2_10m.tif", s2_10m_patch, patch_ds, metadata)
            write_patch(date_dir / "s2_20m.tif", s2_20m_patch, patch_ds, metadata)
            write_coverage(date_dir / "s2_valid.tif", valid, patch_ds, metadata)
            s2_count += 1
            logging.info(
                "Saved Sentinel-2 mosaic for %s from tiles %s (coverage %.2f%%)",
                date_str,
                metadata["s2_source_tiles"],
                coverage * 100,
            )

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
