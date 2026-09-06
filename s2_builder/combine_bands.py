import elevation
import geopandas as gpd
import logging
import numpy as np
import os
import rasterio

from pathlib import Path
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.features import rasterize
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import Window


os.environ["PROJ_LIB"] = "/home/jovyan/nfs/mgatti/python/landslides-detection/.venv/lib/python3.11/site-packages/pyproj/proj_dir/share/proj"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

IMAGES_PATH = Path("/home/jovyan/nfs/tesista3/landslide-detection/Landslides/Sentinel/images/")
ANNOTATIONS_PATH = Path("inventories")

PRODUCTS_10m = ["B02_10m", "B03_10m", "B04_10m", "B08_10m"]
DESCRIPTIONS_10m = ["Band 2 - Blue", "Band 3 - Green", "Band 4 - Red", "Band 8 - NIR"]

PRODUCTS_20m = [
    "B05_20m",
    "B06_20m",
    "B07_20m",
    "B8A_20m",
    "B11_20m",
    "B12_20m"
]

DESCRIPTIONS_20m = [
    "Band 5 - Vegetation red edge",
    "Band 6 - Vegetation red edge",
    "Band 7 - Vegetation red edge",
    "Band 8A - Vegetation red edge",
    "Band 11 - SWIR",
    "Band 12 - SWIR"
]

PRODUCTS_60m = ["B01_60m", "B09_60m", "B10_60m"]
DESCRIPTIONS_60m = ["Band 1 - Coastal aerosol", "Band 9 - Water vapour", "Band 10 - SWIR - Cirrus"]

PRODUCT_LEVELS = ["MSIL1C", "MSIL2A"]

annotations = {}

for gpkg_path in ANNOTATIONS_PATH.glob("*.gpkg"):
    inventory_name = gpkg_path.stem
    annotations[inventory_name] = gpkg_path


def transform_bounds(raster_crs, raster_bounds, target_crs="EPSG:4326"):
    """Transform bounds from raster CRS to target CRS."""
    transformer = Transformer.from_crs(raster_crs, target_crs, always_xy=True)
    min_lon, min_lat = transformer.transform(raster_bounds[0], raster_bounds[1])
    max_lon, max_lat = transformer.transform(raster_bounds[2], raster_bounds[3])
    return min_lon, min_lat, max_lon, max_lat


def download_and_clip_dtm(bounds, output_file):
    """Download and clip the DTM using the elevation library."""
    logging.info(f"Clipping DTM for bounds: {bounds}")
    elevation.clip(bounds=bounds, output=output_file.resolve())
    elevation.clean()
    logging.info(f"DTM clipped and saved to {output_file}")


def reproject_to_raster_crs(input_file, output_file, target_crs):
    """Reproject the downloaded DTM to match the original raster CRS."""
    logging.info(f"Reprojecting {input_file} to CRS: {target_crs}")
    with rasterio.open(input_file) as src:
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        new_meta = src.meta.copy()
        new_meta.update(
            {
                "crs": target_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )

        with rasterio.open(output_file, "w", **new_meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=target_crs,
                resampling=Resampling.cubic,
            )
    logging.info(f"Reprojected DTM saved to {output_file}")


def download_dtm(raster_crs, raster_bounds, dtm_filename):
    """Download, clip, and reproject the DTM."""
    try:
        # Transform raster bounds to WGS84 (EPSG:4326)
        wgs84_bounds = transform_bounds(raster_crs, raster_bounds)

        # Download and clip DTM
        temp_file = dtm_filename.with_suffix(".tmp.tif")
        download_and_clip_dtm(wgs84_bounds, temp_file)

        # Reproject DTM to the original raster CRS
        reproject_to_raster_crs(temp_file, dtm_filename, raster_crs)

        # Clean up temporary file
        temp_file.unlink(missing_ok=True)
    except Exception as e:
        logging.error(f"Error downloading or processing DTM: {e}", exc_info=True)


def clip_raster(raster_src, shapefile_path):
    """Clip raster using a shapefile and return the clipped image and metadata."""
    shapefile = gpd.read_file(shapefile_path, layer="area")
    shapes = [
        feature["geometry"] for feature in shapefile.__geo_interface__["features"]
    ]

    out_raster = mask(raster_src, shapes, all_touched=True, crop=False)

    return out_raster[0].squeeze(0)


def get_crop_size(resolution):
    if resolution == 10.0:
        return 10000
    elif resolution == 20.0:
        return 5000
    elif resolution == 60.0:
        return 1667
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")


def read_crop_band(band_path, shapefile_path=None):
    """Read and optionally clip a band to a center crop and mask."""
    with rasterio.open(band_path.as_posix()) as src:
        pixel_size = src.res[0]
        crop_size = get_crop_size(pixel_size)

        start_x = (src.width - crop_size) // 2
        start_y = (src.height - crop_size) // 2
        window = Window(start_x, start_y, crop_size, crop_size)
        transform = src.window_transform(window)

        band_array = src.read(1, window=window)

        if shapefile_path:
            mem_profile = src.profile.copy()
            mem_profile.update({
                "height": crop_size,
                "width": crop_size,
                "transform": transform,
                "count": 1,
                "driver": "GTiff"
            })

            with MemoryFile() as memfile:
                with memfile.open(**mem_profile) as mem:
                    mem.write(band_array, 1)
                    band_array = clip_raster(mem, shapefile_path)

        return band_array, transform, src.profile.copy(), crop_size


def write_combined_raster(band_arrays, transform, profile, output_path, descriptions):
    """Write a multi-band raster with metadata and band descriptions."""
    profile.update({
        "height": band_arrays[0].shape[0],
        "width": band_arrays[0].shape[1],
        "transform": transform,
        "count": len(band_arrays),
        "driver": "GTiff",
        "crs": profile.get("crs") #aggiunto
    })

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path.as_posix(), "w", **profile) as dst:
        for i, arr in enumerate(band_arrays):
            dst.write(arr, i + 1)
            dst.set_band_description(i + 1, descriptions[i])

    logging.info(f"Multi-band raster saved as {output_path}")


def combine_bands(input_bands, band_descriptions, output_path, shapefile_path=None):
    """Read, crop, and stack bands into a single raster."""
    if output_path.exists():
        logging.info(f"Skipping existing file: {output_path}")
        return

    try:
        band_arrays = []
        transform = None
        profile = None

        for band_path in input_bands:
            band_array, band_transform, band_profile, _ = read_crop_band(band_path, shapefile_path)

            if transform is None:
                transform = band_transform
                profile = band_profile

            band_arrays.append(band_array)

        write_combined_raster(band_arrays, transform, profile, output_path, band_descriptions)

    except Exception as e:
        logging.error(f"Failed to combine bands: {e}")


def save_annotation(shapefile_path, meta, output_path):
    """Rasterize polygon shapefile."""
    shapefile = gpd.read_file(shapefile_path, layer="landslides")
    shapes = ((geom, 1) for geom in shapefile.geometry)

    binary_raster = rasterize(
        shapes,
        out_shape=(meta["height"], meta["width"]),
        transform=meta["transform"],
        fill=0,
        dtype=np.uint8,
    )

    ann_meta = meta.copy()
    ann_meta.update({"count": 1})

    with rasterio.open(output_path.as_posix(), "w", **ann_meta) as dst:
        dst.write(binary_raster, 1)

    logging.info(f"Annotation raster saved as {output_path}")


def process_tile_level(tile, product_level, gpkg_path):
    level_dir = tile / product_level
    if not level_dir.exists():
        return

    date_dirs = [d for d in level_dir.iterdir() if d.is_dir()]
    out_tile_dir = tile.parent / "combined" / tile.name / product_level

    for date_dir in date_dirs:
        date_str = date_dir.name.split("_")[2][:8]
        out_date_dir = out_tile_dir / date_str
        out_date_dir.mkdir(parents=True, exist_ok=True)

        process_band_groups(date_dir, out_date_dir, product_level, gpkg_path)
        save_sen2cor_mask(date_dir, out_date_dir)

        if product_level == "MSIL2A":
            generate_dtm_and_annotation(date_str, out_date_dir, gpkg_path, out_tile_dir)


def process_band_groups(date_dir, out_dir, level, gpkg_path):
    date_str = date_dir.name.split("_")[2][:8]
    products_60m = PRODUCTS_60m if level == "MSIL1C" else PRODUCTS_60m[:-1]
    descriptions_60m = DESCRIPTIONS_60m if level == "MSIL1C" else DESCRIPTIONS_60m[:-1]

    band_groups = [
        (PRODUCTS_10m, DESCRIPTIONS_10m, f"{date_str}_10m.tif"),
        (PRODUCTS_20m, DESCRIPTIONS_20m, f"{date_str}_20m.tif"),
        (products_60m, descriptions_60m, f"{date_str}_60m.tif"),
    ]

    for products, descs, fname in band_groups:
        out_file = out_dir / fname
        if out_file.exists():
            logging.info(f"Skipping existing file: {out_file}")
            continue
        try:
            band_paths = [list(date_dir.rglob(f"*{p.split('_')[0]}.jp2"))[0] for p in products]
            combine_bands(band_paths, descs, out_file, gpkg_path)
        except IndexError:
            logging.warning(f"Missing product in {date_dir}: {products}")


def save_sen2cor_mask(date_dir, out_dir):
    scl_file = next(date_dir.rglob("*_SCL_20m.jp2"), None)
    if not scl_file:
        return

    mask_path = out_dir / "sen2cor.tif"
    if mask_path.exists():
        return

    with rasterio.open(scl_file) as src:
        pixel_size = src.res[0]
        crop_size = get_crop_size(pixel_size)
        window = Window(
            (src.width - crop_size) // 2,
            (src.height - crop_size) // 2,
            crop_size, crop_size
        )
        transform = src.window_transform(window)
        profile = src.profile.copy()
        profile.update({
            "height": crop_size, "width": crop_size,
            "transform": transform, "compress": "lzw", "driver": "GTiff"
        })
        with rasterio.open(mask_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                dst.write(src.read(i, window=window), i)
    logging.info(f"Saved Sen2Cor mask: {mask_path}")


def generate_dtm_and_annotation(date_str, out_dir, gpkg_path, out_tile_dir):
    raster_path = out_dir / f"{date_str}_10m.tif"
    if not raster_path.exists():
        logging.warning(f"Missing raster for DTM/annotation: {raster_path}")
        return

    with rasterio.open(raster_path) as src:
        crs, bounds, meta = src.crs, src.bounds, src.meta

    dtm_path = out_tile_dir / "dtm.tif"
    if not dtm_path.exists():
        download_dtm(crs, bounds, dtm_path)
    else:
        logging.info("DTM already exists.")

    ann_path = out_tile_dir / "landslides.tif"
    if not ann_path.exists():
        save_annotation(gpkg_path, meta, ann_path)
    else:
        logging.info("Annotation already exists.")


def process_inventory(inventory_path):
    gpkg_path = ANNOTATIONS_PATH / f"{inventory_path.name}.gpkg"
    if not gpkg_path.exists():
        logging.warning(f"No annotation file found for: {inventory_path.name}")
        return

    for tile in inventory_path.glob("T[0-9][0-9]*"):
        for level in PRODUCT_LEVELS:
            process_tile_level(tile, level, gpkg_path)


if __name__ == "__main__":
    for inventory in IMAGES_PATH.iterdir():
        if inventory.is_dir():
            logging.info(f"Processing: {inventory.name}")
            process_inventory(inventory)

    logging.info("All inventories processed.")