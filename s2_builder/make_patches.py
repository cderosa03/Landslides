from collections import defaultdict
import logging
import numpy as np
import os
import argparse
import csv
import shutil
import tempfile
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
PS_BASE = Path(os.getenv("PS_PATCHES_PATH", PROJECT_ROOT / "Landslides" / "PlanetScope" / "patches"))
S2_BASE = Path(os.getenv("S2_IMAGES_PATH", PROJECT_ROOT / "Landslides" / "Sentinel" / "images"))
S2_PRODUCT_LEVEL = "MSIL2A"
S2_COMBINED_ROOT = None  # optional isolated combined output
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
    s2_event_dir = (S2_COMBINED_ROOT / event) if S2_COMBINED_ROOT is not None else (S2_BASE / event / "combined")
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




def parse_cli():
    parser = argparse.ArgumentParser(description="Generate clean Sentinel-2 patches in a separate output directory")
    parser.add_argument("--patches-root", type=Path, default=PS_BASE)
    parser.add_argument("--s2-images-root", type=Path, default=S2_BASE)
    parser.add_argument("--s2-combined-root", type=Path, default=None, help="Use staged combined output (event/tile/MSIL2A/date) instead of original combined/")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "s2_patches_candidate")
    parser.add_argument("--events", nargs="+", default=EVENTS)
    parser.add_argument("--limit-patches", type=int, default=0, help="0 means all")
    parser.add_argument("--apply", action="store_true", help="Actually create outputs; otherwise dry-run")
    parser.add_argument("--resume", action="store_true", help="Skip complete date outputs from prior runs")
    parser.add_argument("--min-valid-coverage", type=float, default=MIN_VALID_COVERAGE)
    parser.add_argument("--report", type=Path, default=Path("s2_patches_build_report.csv"))
    args = parser.parse_args()
    if not 0 <= args.min_valid_coverage <= 1:
        parser.error("--min-valid-coverage must be in [0,1]")
    if args.limit_patches < 0:
        parser.error("--limit-patches must be >= 0")
    # Never write inside existing PlanetScope patch tree.
    src = args.patches_root.resolve()
    out = args.output_root.resolve()
    if out == src or src in out.parents or out in src.parents:
        parser.error("--output-root must be separate from --patches-root")
    return args


def complete_date(folder):
    paths = [folder / name for name in ("s2_10m.tif", "s2_20m.tif", "s2_valid.tif")]
    if not all(p.is_file() and p.stat().st_size > 0 for p in paths):
        return False
    try:
        with rasterio.open(paths[0]) as a, rasterio.open(paths[1]) as b, rasterio.open(paths[2]) as v:
            return (a.crs == b.crs == v.crs and a.transform == b.transform == v.transform
                    and a.width == b.width == v.width and a.height == b.height == v.height
                    and v.count == 1)
    except Exception:
        return False


def write_date_atomically(folder, image10, image20, valid, patch_ds, metadata):
    folder.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=".building_", dir=folder.parent))
    try:
        write_patch(temp_dir / "s2_10m.tif", image10, patch_ds, metadata)
        write_patch(temp_dir / "s2_20m.tif", image20, patch_ds, metadata)
        write_coverage(temp_dir / "s2_valid.tif", valid, patch_ds, metadata)
        if not complete_date(temp_dir):
            raise RuntimeError(f"Incomplete output: {temp_dir}")
        if folder.exists():
            raise FileExistsError(f"Output already exists: {folder}; choose a clean --output-root")
        temp_dir.rename(folder)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


def process_one_patch(patch_dir, event, args, report_writer):
    reference = patch_dir / "pre.tif"
    if not reference.is_file():
        report_writer.writerow([event, patch_dir.name, "", "missing_pre", ""])
        return
    with rasterio.open(reference) as patch_ds:
        tiles = find_overlapping_s2_tiles(event, patch_ds)
        if not tiles:
            report_writer.writerow([event, patch_dir.name, "", "no_overlapping_tiles", ""])
            return
        dates = group_images_by_date(tiles)
        for date_str, sources in dates.items():
            target = args.output_root / event / patch_dir.name / "s2" / date_str
            if args.resume and complete_date(target):
                report_writer.writerow([event, patch_dir.name, date_str, "already_complete", ""])
                continue
            # Never silently overwrite partial or old output.
            if target.exists():
                report_writer.writerow([event, patch_dir.name, date_str, "output_exists", ""])
                continue
            if not args.apply:
                report_writer.writerow([event, patch_dir.name, date_str, "would_generate", ""])
                continue
            try:
                image10, image20, valid = mosaic_s2_images(sources, patch_ds)
                if image10 is None:
                    report_writer.writerow([event, patch_dir.name, date_str, "no_usable_source", ""])
                    continue
                coverage = float(valid.mean())
                if coverage < args.min_valid_coverage:
                    report_writer.writerow([event, patch_dir.name, date_str, "low_coverage", f"{coverage:.6f}"])
                    continue
                metadata = {
                    **mosaic_metadata(date_str, sources),
                    "s2_valid_coverage": f"{coverage:.6f}",
                    "s2_invalid_fraction": f"{1 - coverage:.6f}",
                    "s2_invalid_scl_classes": ",".join(map(str, sorted(INVALID_SCL_CLASSES))),
                }
                write_date_atomically(target, image10, image20, valid, patch_ds, metadata)
                report_writer.writerow([event, patch_dir.name, date_str, "generated", f"{coverage:.6f}"])
            except Exception as exc:
                logging.exception("Failed %s/%s/%s", event, patch_dir.name, date_str)
                report_writer.writerow([event, patch_dir.name, date_str, "error", str(exc)])


def main():
    args = parse_cli()
    global S2_BASE, S2_COMBINED_ROOT
    S2_BASE = args.s2_images_root
    S2_COMBINED_ROOT = args.s2_combined_root
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Flush incrementally; errors never destroy existing patch data.
    with args.report.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["event", "patch_id", "date", "status", "coverage_or_error"])
        for event in args.events:
            event_dir = args.patches_root / event
            if not event_dir.is_dir():
                logging.error("Missing event directory %s", event_dir)
                writer.writerow([event, "", "", "missing_event_directory", str(event_dir)])
                continue
            patches = sorted((p for p in event_dir.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))
            if args.limit_patches:
                patches = patches[:args.limit_patches]
            logging.info("%s: %d patches (%s)", event, len(patches), "APPLY" if args.apply else "DRY RUN")
            for index, patch in enumerate(patches, 1):
                process_one_patch(patch, event, args, writer)
                stream.flush()
                if index % 100 == 0 or index == len(patches):
                    logging.info("%s: %d/%d patches", event, index, len(patches))
    logging.info("Report saved: %s", args.report)


if __name__ == "__main__":
    main()
