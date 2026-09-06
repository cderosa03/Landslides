#!/usr/bin/env python3
"""
PlanetScope landslide dataset builder (PEP 8 compliant).

Key features:
- Windowed, memory-safe patch generation (no giant view_as_windows tensors).
- Optional skipping of negative patches (default: skip).
- Threaded quad downloads with resilient pagination and retries.
- Robust alpha-band handling for PlanetScope pre/post rasters.
- CLI configurability (patch size, stride, cloud threshold, workers, etc.).
- Native-grid "wide context" exports per patch: dem_wide, slope_wide, aspect_wide.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import elevation
import geopandas as gpd
import numpy as np
import rasterio
from osgeo import gdal
from rasterio.features import rasterize
from rasterio.merge import merge
from rasterio.transform import Affine
from rasterio.warp import (
    Resampling,
    calculate_default_transform,
    reproject,
    transform_bounds,
)
from rasterio.windows import Window
from requests import Session
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

# ======================================================================================
# Defaults / Config
# ======================================================================================

BASE_DIR = Path(
    os.getenv(
        "PLANET_LS_BASE",
        "/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope",
    )
)
IMAGES_DIR = BASE_DIR / "images"
PATCHES_DIR = BASE_DIR / "patches"
ANNOTATIONS_PATH = Path("./inventories")
KEY_PATH = Path("api_key.json")
API_URL = "https://api.planet.com/basemaps/v1/mosaics"

# IO / GDAL
TILE_BLOCK = 512
COMPRESS = "lzw"
BIGTIFF = "YES"

# HTTP
REQ_TIMEOUT = 60
RETRY_TOTAL = 5
RETRY_BACKOFF = 0.8

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
LOG = logging.getLogger("planet-pipeline")

# ======================================================================================
# Types
# ======================================================================================


@dataclass(frozen=True)
class Inventory:
    """Inventory descriptor: dataset name and event dates."""

    name: str
    dates: List[datetime]


INVENTORIES: List[Inventory] = [
    Inventory("Lombok2018", [datetime(2018, 8, 5), datetime(2018, 8, 19)]),
    Inventory(
        "Philippines2019",
        [
            datetime(2019, 10, 16),
            datetime(2019, 10, 29),
            datetime(2019, 10, 31),
            datetime(2019, 12, 15),
        ],
    ),
    Inventory("Michoacan2022", [datetime(2022, 9, 19)]),
    Inventory("EmiliaRomagna2023", [datetime(2023, 5, 16), datetime(2023, 5, 17)]),
]

# ======================================================================================
# HTTP / Auth
# ======================================================================================


def load_api_key() -> str:
    """Load Planet API key from env or api_key.json."""

    env_key = os.getenv("PLANET_API_KEY")
    if env_key:
        return env_key.strip()

    if KEY_PATH.exists():
        with open(KEY_PATH, "r", encoding="utf-8") as fobj:
            return json.load(fobj)["apiKey"]

    raise RuntimeError(
        "No API key found. Set PLANET_API_KEY or provide api_key.json with "
        "{'apiKey': '...'}.")


def make_session() -> Session:
    """Configure an authenticated, resilient requests session."""

    api_key = load_api_key()
    session = Session()
    session.auth = (api_key, "")
    retries = Retry(
        total=RETRY_TOTAL,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET", "HEAD"]),
        raise_on_status=False,
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    session.headers.update({"User-Agent": "planet-lsm/2.0"})
    return session


# ======================================================================================
# Helpers
# ======================================================================================


def shapefile_to_bbox(inv: Inventory, target_epsg: int = 4326) -> Tuple[float, ...]:
    """Read <name>.gpkg layer 'area' and return bbox in EPSG:4326 (lon/lat)."""

    gpkg_path = ANNOTATIONS_PATH / f"{inv.name}.gpkg"
    gdf = gpd.read_file(gpkg_path, layer="area")
    gdf = gdf.to_crs(epsg=target_epsg)
    return tuple(gdf.total_bounds)


def get_previous_month(month: int, year: int) -> Tuple[int, int]:
    """Return previous month and year."""

    return (12, year - 1) if month == 1 else (month - 1, year)


def get_next_month(month: int, year: int) -> Tuple[int, int]:
    """Return next month and year."""

    return (1, year + 1) if month == 12 else (month + 1, year)


def write_big_geotiff(
    out_path: Path,
    data: np.ndarray,
    base_meta: dict,
    *,
    block: int = TILE_BLOCK,
    compress: str = COMPRESS,
    bigtiff: str = BIGTIFF,
) -> None:
    """Write tiled, compressed (optionally multi-band) GeoTIFF."""

    meta = base_meta.copy()
    meta.update({
        "driver": "GTiff",
        "compress": compress,
        "dtype": data.dtype,
    })

    if data.ndim == 2:
        meta.update({"count": 1, "height": data.shape[0], "width": data.shape[1]})
    else:
        meta.update({
            "count": data.shape[0],
            "height": data.shape[1],
            "width": data.shape[2],
        })

    meta.setdefault("tiled", True)
    meta.setdefault("blockxsize", block)
    meta.setdefault("blockysize", block)
    meta.setdefault("BIGTIFF", bigtiff)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with rasterio.open(tmp, "w", **meta) as dst:
        if data.ndim == 2:
            dst.write(data, 1)
        else:
            dst.write(data)
    tmp.replace(out_path)


def window_bounds(src: rasterio.io.DatasetReader, window: Window) -> Tuple[float, ...]:
    """Return (minx, miny, maxx, maxy) in dataset CRS for a Window."""

    transform = rasterio.windows.transform(window, src.transform)
    height, width = int(window.height), int(window.width)

    x0, y0 = transform * (0, 0)
    x1, y1 = transform * (width, height)

    minx, maxx = (x0, x1) if x0 <= x1 else (x1, x0)
    miny, maxy = (y1, y0) if y1 <= y0 else (y0, y1)
    return minx, miny, maxx, maxy


def has_alpha_band(src: rasterio.io.DatasetReader) -> bool:
    """Return True if dataset has an alpha band."""

    try:
        return any(ci.name.lower() == "alpha" for ci in src.colorinterp)
    except Exception:
        return False


def drop_alpha_if_present(arr: np.ndarray, src: rasterio.io.DatasetReader) -> np.ndarray:
    """Drop a true alpha band or a constant last band from (C, H, W) arrays."""

    if arr.ndim != 3 or arr.shape[0] < 4:
        return arr

    if has_alpha_band(src):
        for bidx, ci in enumerate(src.colorinterp, start=1):
            if ci.name.lower() == "alpha":
                return np.delete(arr, bidx - 1, axis=0)

    last = arr[-1]
    if np.all(last == last.flat[0]) and last.flat[0] in (255, 0):
        return arr[:-1]

    return arr


def download_stream(session: Session, url: str, out_path: Path) -> None:
    """Stream a URL to disk with retries already handled by the session."""

    with session.get(url, stream=True, timeout=REQ_TIMEOUT) as resp:
        resp.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(out_path.suffix + ".part")
        with open(tmp, "wb") as fobj:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                if chunk:
                    fobj.write(chunk)
        tmp.replace(out_path)


# ======================================================================================
# Planet basemaps
# ======================================================================================


def _fetch_mosaic_id(session: Session, mosaic_name: str) -> Optional[str]:
    """Resolve a basemap mosaic name to its id."""

    res = session.get(API_URL, params={"name__is": mosaic_name}, timeout=REQ_TIMEOUT)
    res.raise_for_status()
    mosaics = res.json().get("mosaics", [])
    return mosaics[0]["id"] if mosaics else None


def _iter_quads(session: Session, mosaic_id: str, bbox_str: str) -> Iterable[dict]:
    """Yield quad items overlapping the bbox for a given mosaic id."""

    url = f"{API_URL}/{mosaic_id}/quads"
    params = {"bbox": bbox_str, "minimal": True}

    while url:
        res = session.get(url, params=params, timeout=REQ_TIMEOUT)
        res.raise_for_status()
        data = res.json()
        for item in data.get("items", []):
            yield item
        params = None
        url = data.get("_links", {}).get("_next")


def download_basemaps(
    inventories: Sequence[Inventory], base_dir: Path, *, workers: int = 8
) -> None:
    """For each inventory, download pre/post month quads overlapping the bbox."""

    session = make_session()

    for inv in inventories:
        dates = inv.dates
        name = inv.name
        month, year = dates[0].month, dates[0].year
        month_before, year_before = get_previous_month(month, year)
        month_after, year_after = get_next_month(dates[-1].month, dates[-1].year)

        area_bbox = shapefile_to_bbox(inv)
        bbox_str = ",".join(map(str, area_bbox))

        for tag, year_i, month_i in [
            ("pre", year_before, month_before),
            ("post", year_after, month_after),
        ]:
            mosaic_name = f"global_monthly_{year_i}_{month_i:02}_mosaic"
            LOG.info("[%s] Resolving mosaic '%s' ...", name, mosaic_name)
            try:
                mosaic_id = _fetch_mosaic_id(session, mosaic_name)
                if not mosaic_id:
                    LOG.warning("[%s] No mosaic id for '%s' (skipping).", name, mosaic_name)
                    continue
            except Exception as exc:  # noqa: BLE001
                LOG.exception(
                    "[%s] Error retrieving mosaic '%s': %s",
                    name,
                    mosaic_name,
                    exc,
                )
                continue

            output_dir = base_dir / name / f"{tag}_quads"
            output_dir.mkdir(parents=True, exist_ok=True)

            items = list(_iter_quads(session, mosaic_id, bbox_str))
            if not items:
                LOG.warning(
                    "[%s] No quads found for %s %04d-%02d.", name, tag, year_i, month_i
                )
                continue

            futures = []
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for quad in items:
                    url = quad["_links"]["download"]
                    fpath = output_dir / f"{quad['id']}_{tag}.tiff"
                    if fpath.exists():
                        continue
                    futures.append(ex.submit(download_stream, session, url, fpath))

                for _ in tqdm(
                    as_completed(futures), total=len(futures), desc=f"{name} - {tag}"
                ):
                    pass


def merge_quads(folder: Path, tag: str) -> None:
    """Merge <folder>/<tag>_quads/*.tiff into <folder>/<tag>_merged.tif."""

    input_dir = folder / f"{tag}_quads"
    output_path = folder / f"{tag}_merged.tif"

    if output_path.exists():
        LOG.info("%s already exists, skipping merge.", output_path.name)
        return

    tiff_files = [fp for fp in input_dir.glob("*.tiff") if not fp.name.startswith("._")]
    if not tiff_files:
        LOG.warning("No TIFF files found in %s, skipping merge.", input_dir)
        return

    src_files = [rasterio.open(fp) for fp in tiff_files]
    try:
        mosaic, out_trans = merge(src_files)
        out_meta = src_files[0].meta.copy()
    finally:
        for src in src_files:
            src.close()

    out_meta.update({"transform": out_trans})
    write_big_geotiff(output_path, mosaic, out_meta)
    LOG.info("Saved merged file to %s", output_path)


# ======================================================================================
# DEM + derivatives (aligned + native)
# ======================================================================================


def bounds_in_epsg4326(tif_path: Path) -> Tuple[float, ...]:
    """Return bounds in EPSG:4326 for a given raster (transform if needed)."""

    with rasterio.open(tif_path) as src:
        bounds = src.bounds
        if src.crs != "EPSG:4326":
            return transform_bounds(src.crs, "EPSG:4326", *bounds)
        return bounds.left, bounds.bottom, bounds.right, bounds.top


def download_dem(inv: Inventory, base_dir: Path) -> None:
    """Download DEM and emit both aligned-grid and native-grid versions."""

    name = inv.name
    folder = base_dir / name
    pre_tif = folder / "pre_merged.tif"

    if not pre_tif.exists():
        LOG.warning("No pre_merged.tif found in %s, skipping DEM.", folder)
        return

    dem_aligned = folder / "dem.tif"
    dem_native = folder / "dem_native.tif"
    if dem_aligned.exists() and dem_native.exists():
        LOG.info("DEM (aligned + native) already exist for %s, skipping.", name)
        return

    LOG.info("Downloading DEM for %s...", name)
    bounds = bounds_in_epsg4326(pre_tif)

    raw_path = folder / "dem_raw.tif"
    with open(os.devnull, "w", encoding="utf-8") as fnull:
        with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
            elevation.clip(bounds=bounds, output=raw_path.resolve())
            elevation.clean()

    with rasterio.open(pre_tif) as ref:
        target_crs = ref.crs
        ref_transform = ref.transform
        ref_width, ref_height = ref.width, ref.height

    # 1) Write dem_native.tif (CRS matched, native grid)
    with rasterio.open(raw_path) as src:
        transform_native, w_native, h_native = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )
        meta_native = src.meta.copy()
        meta_native.update(
            crs=target_crs,
            transform=transform_native,
            width=w_native,
            height=h_native,
            driver="GTiff",
            compress=COMPRESS,
            tiled=True,
            blockxsize=TILE_BLOCK,
            blockysize=TILE_BLOCK,
            BIGTIFF=BIGTIFF,
        )
        tmp_native = dem_native.with_suffix(".tif.part")
        with rasterio.open(tmp_native, "w", **meta_native) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform_native,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        tmp_native.replace(dem_native)

    # 2) Write dem.tif aligned to pre_merged.tif grid
    with rasterio.open(raw_path) as src:
        meta_aligned = src.meta.copy()
        meta_aligned.update(
            crs=target_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
            driver="GTiff",
            compress=COMPRESS,
            tiled=True,
            blockxsize=TILE_BLOCK,
            blockysize=TILE_BLOCK,
            BIGTIFF=BIGTIFF,
        )
        tmp_aligned = dem_aligned.with_suffix(".tif.part")
        with rasterio.open(tmp_aligned, "w", **meta_aligned) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref_transform,
                dst_crs=target_crs,
                resampling=Resampling.bilinear,
            )
        tmp_aligned.replace(dem_aligned)

    raw_path.unlink(missing_ok=True)
    LOG.info("DEM saved: %s (aligned), %s (native)", dem_aligned.name, dem_native.name)


def build_slope_aspect(inv: Inventory, base_dir: Path) -> None:
    """Compute slope/aspect for aligned and native DEMs using GDAL."""

    name = inv.name
    folder = base_dir / name

    dem_aligned = folder / "dem.tif"
    dem_native = folder / "dem_native.tif"

    if not dem_aligned.exists() or not dem_native.exists():
        LOG.warning("DEM files missing for %s; cannot compute slope/aspect.", name)
        return

    outputs = [
        (dem_aligned, folder / "slope.tif", folder / "aspect.tif"),
        (dem_native, folder / "slope_native.tif", folder / "aspect_native.tif"),
    ]

    for dem_path, slope_path, aspect_path in outputs:
        if slope_path.exists() and aspect_path.exists():
            continue

        LOG.info(
            "Computing slope/aspect for %s -> %s / %s",
            dem_path.name,
            slope_path.name,
            aspect_path.name,
        )
        gdal.DEMProcessing(
            str(slope_path),
            str(dem_path),
            "slope",
            format="GTiff",
            slopeFormat="degree",
            computeEdges=True,
        )
        gdal.DEMProcessing(
            str(aspect_path),
            str(dem_path),
            "aspect",
            format="GTiff",
            computeEdges=True,
        )


# ======================================================================================
# Rasterization
# ======================================================================================


def rasterize_layer_to_match(
    gpkg_path: Path,
    layer_name: str,
    ref_tif: Path,
    out_path: Path,
    burn_value: int = 1,
) -> None:
    """Rasterize a GeoPackage layer to match a reference raster grid/CRS."""

    if out_path.exists():
        return

    try:
        gdf = gpd.read_file(gpkg_path, layer=layer_name)
    except ValueError as err:  # noqa: B904
        LOG.error("Error reading layer '%s' from %s: %s", layer_name, gpkg_path, err)
        return

    with rasterio.open(ref_tif) as src:
        transform = src.transform
        out_shape = (src.height, src.width)
        raster_crs = src.crs

    if gdf.empty:
        mask = np.zeros(out_shape, dtype="uint8")
    else:
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
        shapes = (
            (geom, burn_value)
            for geom in gdf.geometry
            if geom is not None and not geom.is_empty
        )
        mask = rasterize(
            shapes=shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
        )

    base_meta = {"crs": raster_crs, "transform": transform}
    write_big_geotiff(out_path, mask, base_meta)


def rasterize_landslides(inv: Inventory, base_dir: Path) -> None:
    """Rasterize 'landslides' layer onto pre_merged.tif grid."""

    name = inv.name
    folder = base_dir / name
    pre_tif = folder / "pre_merged.tif"
    out_raster = folder / "landslides.tif"

    if out_raster.exists():
        LOG.info("Landslides raster already exists for %s, skipping.", name)
        return
    if not pre_tif.exists():
        LOG.error("pre_merged.tif not found for %s, cannot rasterize landslides.", name)
        return

    gpkg_path = ANNOTATIONS_PATH / f"{name}.gpkg"
    if not gpkg_path.exists():
        LOG.error("GeoPackage not found: %s", gpkg_path)
        return

    rasterize_layer_to_match(gpkg_path, "landslides", pre_tif, out_raster)
    LOG.info("Rasterized landslides saved to %s", out_raster)


def rasterize_area(inv: Inventory, base_dir: Path) -> None:
    """Rasterize 'area' layer onto pre_merged.tif grid (burn value 1)."""

    name = inv.name
    folder = base_dir / name
    pre_tif = folder / "pre_merged.tif"
    out_raster = folder / "area_mask.tif"

    if out_raster.exists():
        LOG.info("Area mask already exists for %s, skipping.", name)
        return
    if not pre_tif.exists():
        LOG.error("pre_merged.tif not found for %s, cannot rasterize area.", name)
        return

    gpkg_path = ANNOTATIONS_PATH / f"{name}.gpkg"
    if not gpkg_path.exists():
        LOG.error("GeoPackage not found: %s", gpkg_path)
        return

    rasterize_layer_to_match(gpkg_path, "area", pre_tif, out_raster, burn_value=1)
    LOG.info("Area mask saved to %s", out_raster)


def rasterize_clouds(inv: Inventory, base_dir: Path) -> None:
    """Rasterize cloud masks to pre_merged.tif grid (layers 'pre' and 'post')."""

    name = inv.name
    folder = base_dir / name
    pre_tif = folder / "pre_merged.tif"

    if not pre_tif.exists():
        LOG.error("pre_merged.tif not found for %s, cannot rasterize clouds.", name)
        return

    gpkg_clouds = ANNOTATIONS_PATH / f"{name}_clouds.gpkg"
    if not gpkg_clouds.exists():
        LOG.warning("Clouds GeoPackage not found: %s (skipping clouds).", gpkg_clouds)
        return

    out_pre = folder / "cloud_mask_pre.tif"
    out_post = folder / "cloud_mask_post.tif"

    if out_pre.exists() and out_post.exists():
        LOG.info("Cloud rasters already exist for %s, skipping.", name)
        return

    rasterize_layer_to_match(gpkg_clouds, "pre", pre_tif, out_pre)
    rasterize_layer_to_match(gpkg_clouds, "post", pre_tif, out_post)

    if out_pre.exists() and out_post.exists():
        LOG.info("Cloud masks saved to %s / %s", out_pre.name, out_post.name)


# ======================================================================================
# Alignment & patching
# ======================================================================================


def check_alignment(paths: List[Path]) -> bool:
    """Ensure all rasters share the same shape, transform, and CRS."""

    metas = []
    for path in paths:
        with rasterio.open(path) as src:
            metas.append((path.name, src.shape, src.transform, src.crs))

    ok = True
    _, shape0, transform0, crs0 = metas[0]
    for name, shape, transform, crs in metas[1:]:
        if shape != shape0:
            LOG.warning("Shape mismatch: %s %s vs %s", name, shape, shape0)
            ok = False
        if (np.asarray(transform) != np.asarray(transform0)).any():
            LOG.warning("Transform mismatch: %s", name)
            ok = False
        if crs != crs0:
            LOG.warning("CRS mismatch: %s %s vs %s", name, crs, crs0)
            ok = False
    return ok


def grid_from_shape(
    height: int, width: int, patch_size: Tuple[int, int], step: Tuple[int, int]
) -> Tuple[int, int]:
    """Compute number of windows that fit for a given shape/stride."""

    ph, pw = patch_size
    sy, sx = step
    n_rows = (height - ph) // sy + 1
    n_cols = (width - pw) // sx + 1
    return n_rows, n_cols


def iter_windows(
    n_rows: int,
    n_cols: int,
    patch_size: Tuple[int, int],
    step: Tuple[int, int],
    base_transform: Affine,
):
    """Yield (i, j, window, patch_transform) in scan order."""

    ph, pw = patch_size
    sy, sx = step
    for i_idx in range(n_rows):
        for j_idx in range(n_cols):
            x_off, y_off = j_idx * sx, i_idx * sy
            window = Window(x_off, y_off, pw, ph)
            yield i_idx, j_idx, window, rasterio.windows.transform(
                window, base_transform
            )


def _write_geotiff(path: Path, data: np.ndarray, crs, transform) -> None:
    """Write a small (window) GeoTIFF; supports (C, H, W) or (H, W)."""

    if data.ndim == 2:
        count, dtype = 1, data.dtype
        height, width = data.shape
    else:
        count, dtype = data.shape[0], data.dtype
        _, height, width = data.shape

    meta = dict(
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress=COMPRESS,
        tiled=True,
        blockxsize=TILE_BLOCK,
        blockysize=TILE_BLOCK,
        BIGTIFF=BIGTIFF,
    )
    tmp = path.with_suffix(path.suffix + ".part")
    with rasterio.open(tmp, "w", **meta) as dst:
        if count == 1:
            dst.write(data, 1)
        else:
            dst.write(data)
    tmp.replace(path)


def _write_window_from_src(
    src: rasterio.io.DatasetReader, out_path: Path, window: Window, transform
) -> None:
    """Read a window from 'src' and write it to 'out_path' with the transform."""

    data = src.read(window=window)
    meta = src.meta.copy()
    meta.update(
        {
            "height": int(window.height),
            "width": int(window.width),
            "transform": transform,
            "compress": COMPRESS,
            "tiled": True,
            "blockxsize": TILE_BLOCK,
            "blockysize": TILE_BLOCK,
            "BIGTIFF": BIGTIFF,
        }
    )
    tmp = out_path.with_suffix(out_path.suffix + ".part")
    with rasterio.open(tmp, "w", **meta) as dst:
        dst.write(data)
    tmp.replace(out_path)


def _center_native_window(
    ref_transform: Affine,
    i_idx: int,
    j_idx: int,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    nat_ds: rasterio.io.DatasetReader,
) -> Tuple[Window, Affine]:
    """Compute a native-grid window centered at the Planet patch center."""

    ph, pw = patch_size
    sy, sx = stride

    row = i_idx * sy + ph // 2
    col = j_idx * sx + pw // 2

    cx, cy = ref_transform * (col + 0.5, row + 0.5)

    n_row, n_col = nat_ds.index(cx, cy)
    r0 = int(n_row - ph // 2)
    c0 = int(n_col - pw // 2)

    win = Window(col_off=c0, row_off=r0, width=pw, height=ph)
    wtransform = rasterio.windows.transform(win, nat_ds.transform)
    return win, wtransform


def generate_patches(
    inv: Inventory,
    base_dir: Path,
    patches_dir: Path,
    *,
    patch_size: Tuple[int, int] = (128, 128),
    stride: Tuple[int, int] = (128, 128),
    cloud_skip_threshold: float = 0.0,
    skip_negatives: bool = False,
) -> None:
    """Generate per-patch rasters and native-grid wide-context rasters."""

    name = inv.name
    folder = base_dir / name
    patch_subdir = patches_dir / name

    if patch_subdir.exists():
        LOG.info("Patches already exist for %s, skipping.", name)
        return

    files = {
        "pre": folder / "pre_merged.tif",
        "post": folder / "post_merged.tif",
        "mask": folder / "landslides.tif",
        "dem": folder / "dem.tif",
        "cloud_pre": folder / "cloud_mask_pre.tif",
        "cloud_post": folder / "cloud_mask_post.tif",
        "area": folder / "area_mask.tif",
        "slope": folder / "slope.tif",
        "aspect": folder / "aspect.tif",
        # Native-grid context
        "dem_native": folder / "dem_native.tif",
        "slope_native": folder / "slope_native.tif",
        "aspect_native": folder / "aspect_native.tif",
    }

    missing_required = [k for k in ["pre", "post", "mask", "dem"] if not files[k].exists()]
    if missing_required:
        LOG.error(
            "Missing files for %s: %s. Skipping patch generation.",
            name,
            ", ".join(missing_required),
        )
        return

    to_check = [files["pre"], files["post"], files["mask"]]
    for key in ("cloud_pre", "cloud_post", "area"):
        if files[key].exists():
            to_check.append(files[key])

    if not check_alignment(to_check):
        LOG.warning("Alignment issues detected for %s. Proceeding anyway.", name)

    with rasterio.Env(GDAL_CACHEMAX=1024):
        with rasterio.open(files["pre"]) as pre_ds, \
             rasterio.open(files["post"]) as post_ds, \
             rasterio.open(files["mask"]) as mask_ds, \
             rasterio.open(files["dem"]) as dem_ds, \
             (rasterio.open(files["cloud_pre"]) if files["cloud_pre"].exists() else contextlib.nullcontext(None)) as cloud_pre_ds, \
             (rasterio.open(files["cloud_post"]) if files["cloud_post"].exists() else contextlib.nullcontext(None)) as cloud_post_ds, \
             (rasterio.open(files["area"]) if files["area"].exists() else contextlib.nullcontext(None)) as area_ds, \
             (rasterio.open(files["slope"]) if files["slope"].exists() else contextlib.nullcontext(None)) as slope_ds, \
             (rasterio.open(files["aspect"]) if files["aspect"].exists() else contextlib.nullcontext(None)) as aspect_ds, \
             (rasterio.open(files["dem_native"]) if files["dem_native"].exists() else contextlib.nullcontext(None)) as dem_nat_ds, \
             (rasterio.open(files["slope_native"]) if files["slope_native"].exists() else contextlib.nullcontext(None)) as slp_nat_ds, \
             (rasterio.open(files["aspect_native"]) if files["aspect_native"].exists() else contextlib.nullcontext(None)) as asp_nat_ds:

            height, width = mask_ds.shape
            n_rows, n_cols = grid_from_shape(height, width, patch_size, stride)
            total_patches = n_rows * n_cols

            patch_subdir.mkdir(parents=True, exist_ok=True)
            LOG.info("Generating patches for %s (total possible: %d)", name, total_patches)

            saved = 0
            desc = f"{name} patches"
            with tqdm(total=total_patches, desc=desc) as pbar:
                for i_idx, j_idx, window, patch_transform in iter_windows(
                    n_rows, n_cols, patch_size, stride, pre_ds.transform
                ):
                    pre = pre_ds.read(window=window)
                    pre = drop_alpha_if_present(pre, pre_ds)

                    post = post_ds.read(window=window)
                    post = drop_alpha_if_present(post, post_ds)

                    mask = mask_ds.read(1, window=window).astype("uint8")

                    if area_ds is not None:
                        area = area_ds.read(1, window=window).astype("uint8")
                        if area.sum() == 0:
                            pbar.update(1)
                            continue
                        mask = np.where(area == 1, mask, 0)
                        pre = np.where(area[None, ...] == 1, pre, np.nan)
                        post = np.where(area[None, ...] == 1, post, np.nan)

                    if np.isnan(pre).any() or np.isnan(post).any():
                        pbar.update(1)
                        continue

                    if (
                        cloud_pre_ds is not None
                        and cloud_post_ds is not None
                        and cloud_skip_threshold is not None
                    ):
                        cpre = cloud_pre_ds.read(1, window=window).astype("uint8")
                        cpost = cloud_post_ds.read(1, window=window).astype("uint8")
                        frac_pre = cpre.mean()
                        frac_post = cpost.mean()
                        if (frac_pre > cloud_skip_threshold) or (
                            frac_post > cloud_skip_threshold
                        ):
                            pbar.update(1)
                            continue

                    if skip_negatives and mask.max() == 0:
                        pbar.update(1)
                        continue

                    saved += 1
                    out_dir = patch_subdir / str(saved)
                    out_dir.mkdir(parents=True, exist_ok=True)

                    _write_geotiff(out_dir / "pre.tif", pre, pre_ds.crs, patch_transform)
                    _write_geotiff(out_dir / "post.tif", post, post_ds.crs, patch_transform)
                    _write_geotiff(out_dir / "mask.tif", mask, mask_ds.crs, patch_transform)

                    _write_window_from_src(dem_ds, out_dir / "dem.tif", window, patch_transform)
                    if slope_ds is not None:
                        _write_window_from_src(
                            slope_ds, out_dir / "slope.tif", window, patch_transform
                        )
                    if aspect_ds is not None:
                        _write_window_from_src(
                            aspect_ds, out_dir / "aspect.tif", window, patch_transform
                        )

                    # Native wide context rasters (if available)
                    if dem_nat_ds is not None:
                        win_nat, wtransform_nat = _center_native_window(
                            pre_ds.transform, i_idx, j_idx, patch_size, stride, dem_nat_ds
                        )

                        def _fill_for(ds: rasterio.io.DatasetReader):
                            is_float = np.issubdtype(np.dtype(ds.dtypes[0]), np.floating)
                            return ds.nodata if ds.nodata is not None else (np.nan if is_float else 0)

                        dem_wide = dem_nat_ds.read(
                            1, window=win_nat, boundless=True, fill_value=_fill_for(dem_nat_ds)
                        )
                        _write_geotiff(
                            out_dir / "dem_wide.tif", dem_wide, dem_nat_ds.crs, wtransform_nat
                        )

                        if slp_nat_ds is not None:
                            slp_wide = slp_nat_ds.read(
                                1, window=win_nat, boundless=True, fill_value=_fill_for(slp_nat_ds)
                            )
                            _write_geotiff(
                                out_dir / "slope_wide.tif", slp_wide, slp_nat_ds.crs, wtransform_nat
                            )

                        if asp_nat_ds is not None:
                            asp_wide = asp_nat_ds.read(
                                1, window=win_nat, boundless=True, fill_value=_fill_for(asp_nat_ds)
                            )
                            _write_geotiff(
                                out_dir / "aspect_wide.tif", asp_wide, asp_nat_ds.crs, wtransform_nat
                            )

                    pbar.update(1)

            LOG.info("Saved %d patches for %s in %s", saved, name, patch_subdir)


# ======================================================================================
# Orchestration / CLI
# ======================================================================================


def run_all(
    inventories: Sequence[Inventory],
    images_dir: Path,
    patches_dir: Path,
    *,
    workers: int,
    patch_size: Tuple[int, int],
    stride: Tuple[int, int],
    cloud_skip_threshold: float,
    skip_negatives: bool,
) -> None:
    """Full pipeline execution for all inventories."""

    images_dir.mkdir(parents=True, exist_ok=True)
    patches_dir.mkdir(parents=True, exist_ok=True)

    download_basemaps(inventories, images_dir, workers=workers)

    for inv in inventories:
        name = inv.name
        folder = images_dir / name
        LOG.info("%s", "=" * 40)
        LOG.info("Processing %s", name)
        LOG.info("%s", "=" * 40)

        LOG.info("Merging quads:")
        for tag in ["pre", "post"]:
            LOG.info("- Merging %s quads", tag)
            merge_quads(folder, tag)

        LOG.info("Downloading DEM:")
        download_dem(inv, images_dir)

        LOG.info("Building slope/aspect:")
        build_slope_aspect(inv, images_dir)

        LOG.info("Rasterizing landslides:")
        rasterize_landslides(inv, images_dir)

        rasterize_area(inv, images_dir)

        LOG.info("Rasterizing clouds:")
        rasterize_clouds(inv, images_dir)

        LOG.info("Generating patches:")
        generate_patches(
            inv,
            images_dir,
            patches_dir,
            patch_size=patch_size,
            stride=stride,
            cloud_skip_threshold=cloud_skip_threshold,
            skip_negatives=skip_negatives,
        )

        LOG.info("Completed processing for %s", name)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="PlanetScope landslide dataset builder"
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=BASE_DIR,
        help="Base directory for images/patches",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Download workers for quads",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("H", "W"),
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        default=(128, 128),
        metavar=("SY", "SX"),
    )
    parser.add_argument(
        "--cloud-skip-threshold",
        type=float,
        default=0.0,
        help=">fraction to skip (0..1). 0 => any clouds.",
    )
    parser.add_argument(
        "--skip-negatives",
        action="store_true",
        help="Skip patches with no landslide pixels.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""

    args = parse_args(argv)
    images_dir = args.base_dir / "images"
    patches_dir = args.base_dir / "patches"

    try:
        run_all(
            INVENTORIES,
            images_dir,
            patches_dir,
            workers=args.workers,
            patch_size=tuple(args.patch_size),
            stride=tuple(args.stride),
            cloud_skip_threshold=args.cloud_skip_threshold,
            skip_negatives=args.skip_negatives,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.exception("Pipeline failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())