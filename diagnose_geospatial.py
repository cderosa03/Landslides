#!/usr/bin/env python3
"""Reproducible geometry and coverage checks for PlanetScope/Sentinel patches."""

import argparse
import csv
import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform_bounds


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PATCHES_ROOT = Path(
    os.getenv("PS_PATCHES_PATH", PROJECT_ROOT / "PlanetScope" / "patches")
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patches-root", type=Path, default=DEFAULT_PATCHES_ROOT)
    parser.add_argument("--events", nargs="*", help="Events to inspect (default: all)")
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-pixels", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--plots", action="store_true")
    return parser.parse_args()


def discover_patches(root, events):
    event_dirs = [root / event for event in events] if events else sorted(root.iterdir())
    return [
        patch
        for event_dir in event_dirs
        if event_dir.is_dir()
        for patch in sorted(event_dir.iterdir())
        if patch.is_dir() and patch.name.isdigit()
    ]


def sample_patches(patches, max_samples, seed):
    if max_samples < 1 or max_samples >= len(patches):
        return patches
    return sorted(random.Random(seed).sample(patches, max_samples))


def reference_grid_errors(reference, source, tolerance_pixels):
    errors = []
    if reference.crs is None or source.crs is None:
        return ["missing_crs"], None
    if reference.crs != source.crs:
        errors.append("crs")
    if (reference.width, reference.height) != (source.width, source.height):
        errors.append("dimensions")

    resolution = min(abs(reference.transform.a), abs(reference.transform.e))
    transform_error = max(
        abs(reference.transform.a - source.transform.a),
        abs(reference.transform.e - source.transform.e),
        abs(reference.transform.c - source.transform.c),
        abs(reference.transform.f - source.transform.f),
    ) / resolution
    if transform_error > tolerance_pixels:
        errors.append("transform")

    source_bounds = transform_bounds(source.crs, reference.crs, *source.bounds)
    bounds_error = max(
        abs(left - right)
        for left, right in zip(reference.bounds, source_bounds)
    ) / resolution
    if bounds_error > tolerance_pixels:
        errors.append("bounds")
    return errors, max(transform_error, bounds_error)


def axis_errors(dataset):
    if dataset.transform.a <= 0 or dataset.transform.e >= 0:
        return ["unexpected_axis_orientation"]
    center = xy(dataset.transform, 0, 0)
    row, col = rowcol(dataset.transform, *center)
    return [] if (row, col) == (0, 0) else ["pixel_coordinate_roundtrip"]


def inspect_raster(reference_path, source_path, name, tolerance_pixels):
    row = {"raster": name, "path": str(source_path), "coverage": None, "errors": []}
    if not source_path.exists():
        row["errors"] = ["missing"]
        return row

    with rasterio.open(reference_path) as reference, rasterio.open(source_path) as source:
        errors, alignment_error = reference_grid_errors(reference, source, tolerance_pixels)
        errors.extend(axis_errors(source))
        row.update(
            crs=str(source.crs),
            width=source.width,
            height=source.height,
            resolution_x=source.transform.a,
            resolution_y=source.transform.e,
            alignment_error_pixels=alignment_error,
            errors=errors,
        )
        if name == "s2_valid":
            row["coverage"] = float((source.read(1) > 0).mean())
    return row


def make_plot(pre_path, s2_path, mask_path, output_path):
    import matplotlib.pyplot as plt

    with rasterio.open(pre_path) as source:
        planet = source.read([1, 2, 3]).astype(np.float32)
    with rasterio.open(s2_path) as source:
        sentinel = source.read([3, 2, 1]).astype(np.float32)
    with rasterio.open(mask_path) as source:
        mask = source.read(1)

    def rgb(array):
        high = np.percentile(array, 99, axis=(1, 2), keepdims=True)
        return np.moveaxis(np.clip(array / np.maximum(high, 1), 0, 1), 0, -1)

    figure, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for axis, image, title in zip(axes, (rgb(planet), rgb(sentinel)), ("Planet pre", "Sentinel-2")):
        axis.imshow(image)
        axis.contour(mask > 0, colors="yellow", linewidths=0.6)
        axis.set_title(title)
        axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def inspect_patch(patch, tolerance_pixels, plots_dir=None):
    event = patch.parent.name
    reference = patch / "pre.tif"
    rows = []
    for name, source in (
        ("planet_post", patch / "post.tif"),
        ("mask", patch / "mask.tif"),
    ):
        rows.append(inspect_raster(reference, source, name, tolerance_pixels))

    for date_dir in sorted((patch / "s2").glob("*/")) if (patch / "s2").exists() else []:
        for name, filename in (
            ("s2_10m", "s2_10m.tif"),
            ("s2_20m", "s2_20m.tif"),
            ("s2_valid", "s2_valid.tif"),
        ):
            row = inspect_raster(reference, date_dir / filename, name, tolerance_pixels)
            row["date"] = date_dir.name
            rows.append(row)
        if plots_dir and (date_dir / "s2_10m.tif").exists() and (patch / "mask.tif").exists():
            make_plot(
                reference,
                date_dir / "s2_10m.tif",
                patch / "mask.tif",
                plots_dir / f"{event}_{patch.name}_{date_dir.name}.png",
            )

    for row in rows:
        row.update(event=event, patch_id=patch.name, passed=not row["errors"])
        row["errors"] = ";".join(row["errors"])
    return rows


def write_reports(rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "geospatial_report.json"
    csv_path = output_dir / "geospatial_report.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = sorted({field for row in rows for field in row})
    with csv_path.open("w", newline="", encoding="utf-8") as report:
        writer = csv.DictWriter(report, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path, json_path


def main():
    args = parse_args()
    patches = sample_patches(
        discover_patches(args.patches_root, args.events), args.max_samples, args.seed
    )
    if not patches:
        raise RuntimeError(f"No numeric patch directories found in {args.patches_root}")

    plots_dir = args.output_dir / "plots" if args.plots else None
    rows = [
        row
        for patch in patches
        for row in inspect_patch(patch, args.tolerance_pixels, plots_dir)
    ]
    csv_path, json_path = write_reports(rows, args.output_dir)
    failures = [row for row in rows if not row["passed"]]
    logging.info("Wrote %s and %s for %s patches", csv_path, json_path, len(patches))
    if failures:
        raise SystemExit(f"Geospatial diagnostics failed: {len(failures)} invalid rasters")


if __name__ == "__main__":
    main()
