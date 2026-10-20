import argparse
import csv
import json
import logging
import os
import random
import re
from datetime import date, datetime
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform_bounds


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_PATCHES_ROOT = Path(
    os.getenv("PS_PATCHES_PATH", PROJECT_ROOT / "PlanetScope" / "patches")
)

PLANET_DATE_TAGS = (
    "acquired",
    "ACQUIRED",
    "acquisition_date",
    "ACQUISITION_DATE",
    "DATE_ACQUIRED",
    "acquisitionDate",
)
# File di metadati cercati accanto al raster (es. pre.tif -> pre.json).
PLANET_SIDECAR_SUFFIXES = (
    ".json",
    "_metadata.json",
    ".metadata.json",
    "_metadata.xml",
    ".xml",
)
JSON_DATE_KEYS = ("acquired", "acquisition_date", "acquisitionDate", "datetime")

DATE_RE = re.compile(r"(\d{4})-?(\d{2})-?(\d{2})")


# ---------------------------------------------------------------------------
# Argomenti
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--patches-root", type=Path, default=DEFAULT_PATCHES_ROOT)
    parser.add_argument("--events", nargs="*", help="Eventi da controllare (default: tutti)")
    parser.add_argument(
        "--sensor",
        choices=("planet", "sentinel", "both"),
        default="both",
        help="Sensore da controllare (default: both)",
    )
    parser.add_argument(
        "--dates",
        nargs="*",
        help="Date Sentinel-2 da controllare, corrispondenti ai nomi delle cartelle sotto s2/.",
    )
    parser.add_argument(
        "--event-dates",
        type=Path,
        help="File CSV (colonne event,date) o JSON ({evento: data}) con le date degli eventi.",
    )
    parser.add_argument(
        "--event-date",
        action="append",
        default=[],
        metavar="EVENTO=DATA",
        help="Data di un evento, es. alluvione=20240120. Ripetibile.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Numero massimo di patch da campionare; 0 (default) significa tutte.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tolerance-pixels", type=float, default=0.01)
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.01,
        help="Frazione minima di pixel validi (s2_valid) perché una data Sentinel conti (default 0.01).",
    )
    parser.add_argument(
        "--require-planet-dates",
        action="store_true",
        help="Considera errore (non solo avviso) una data PlanetScope non trovata.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("diagnostics"))
    parser.add_argument("--plots", action="store_true")
    parser.add_argument(
        "--sentinel-strict-grid",
        action="store_true",
        help="Richiede che i raster Sentinel coincidano con pre.tif anche per dimensioni/griglia.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Date
# ---------------------------------------------------------------------------

def parse_date(value):
    """Estrae una data da YYYYMMDD, YYYY-MM-DD o stringhe che li contengono."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    for match in DATE_RE.finditer(str(value)):
        try:
            return date(int(match[1]), int(match[2]), int(match[3]))
        except ValueError:
            continue
    return None


def load_event_dates(pairs=None, path=None):
    raw = {}
    if path:
        if not path.is_file():
            raise FileNotFoundError(f"File date eventi non trovato: {path}")
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError(f"{path}: atteso un oggetto JSON {{evento: data}}")
            raw.update(data)
        else:
            with path.open(newline="", encoding="utf-8") as stream:
                reader = csv.DictReader(stream)
                missing = {"event", "date"} - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"{path}: colonne mancanti {sorted(missing)}")
                for record in reader:
                    if record["event"] and record["event"].strip():
                        raw[record["event"].strip()] = record["date"]
    for pair in pairs or []:
        event, sep, value = pair.partition("=")
        if not sep or not event.strip():
            raise ValueError(f"--event-date non valido: {pair!r} (atteso EVENTO=DATA)")
        raw[event.strip()] = value.strip()

    dates = {}
    for event, value in raw.items():
        parsed = parse_date(value)
        if parsed is None:
            raise ValueError(f"Data evento non valida per {event}: {value!r}")
        dates[event] = parsed
    return dates


def classify_date(acquisition, event_date):
    if acquisition is None or event_date is None:
        return "unknown"
    if acquisition < event_date:
        return "pre"
    if acquisition > event_date:
        return "post"
    return "same_day"


def _find_json_date(obj):
    if isinstance(obj, dict):
        for key in JSON_DATE_KEYS:
            if key in obj:
                found = parse_date(obj[key])
                if found:
                    return found, key
        for value in obj.values():
            found = _find_json_date(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_json_date(value)
            if found:
                return found
    return None


def read_planet_date(path):
    """Cerca la data di acquisizione PlanetScope. Restituisce (data, origine)."""
    for suffix in PLANET_SIDECAR_SUFFIXES:
        sidecar = path.with_name(path.stem + suffix)
        if not sidecar.is_file():
            continue
        try:
            text = sidecar.read_text(encoding="utf-8")
            if suffix.endswith(".json"):
                found = _find_json_date(json.loads(text))
                if found:
                    return found[0], f"{sidecar.name}:{found[1]}"
            else:
                match = re.search(
                    r"acquisition(?:Date|DateTime)[^>]*>\s*([^<]+)", text, re.IGNORECASE
                )
                if match and parse_date(match[1]):
                    return parse_date(match[1]), f"{sidecar.name}:xml"
        except Exception as exc:
            logging.warning("Metadati non leggibili in %s: %s", sidecar, exc)

    if path.exists():
        try:
            with rasterio.open(path) as ds:
                tags = ds.tags()
            for key in PLANET_DATE_TAGS:
                if key in tags:
                    found = parse_date(tags[key])
                    if found:
                        return found, f"tag:{key}"
        except Exception as exc:
            logging.warning("Tag non leggibili in %s: %s", path, exc)
    return None, None


# ---------------------------------------------------------------------------
# Ricerca patch
# ---------------------------------------------------------------------------

def discover_patches(root, events=None):
    if not root.is_dir():
        raise FileNotFoundError(f"Cartella delle patch non trovata: {root}")
    event_dirs = [root / event for event in events] if events else sorted(
        p for p in root.iterdir() if p.is_dir()
    )
    patches = []
    for event_dir in event_dirs:
        if not event_dir.is_dir():
            logging.warning("Cartella evento non trovata: %s", event_dir)
            continue
        patches.extend(
            p for p in sorted(event_dir.iterdir())
            if p.is_dir() and p.name.isdigit()
        )
    return patches


def sample_patches(patches, max_samples, seed):
    if max_samples < 0:
        raise ValueError("--max-samples deve essere >= 0")
    if max_samples == 0 or max_samples >= len(patches):
        return sorted(patches)
    return sorted(random.Random(seed).sample(patches, max_samples))


# ---------------------------------------------------------------------------
# Controlli geospaziali
# ---------------------------------------------------------------------------

def base_row(raster, path, date_value=None):
    return {
        "raster": raster,
        "path": str(path),
        "date": date_value,
        "acquisition_date": None,
        "temporal_class": None,
        "date_source": None,
        "coverage": None,
        "errors": [],
        "warnings": [],
        "crs": None,
        "width": None,
        "height": None,
        "resolution_x": None,
        "resolution_y": None,
        "alignment_error_pixels": None,
    }


def axis_errors(dataset):
    errors = []
    transform = dataset.transform
    # Controllo per le comuni griglie north-up non ruotate.
    if transform.a <= 0 or transform.e >= 0:
        errors.append("unexpected_axis_orientation")
    try:
        x, y = xy(transform, 0, 0, offset="center")
        row, col = rowcol(transform, x, y)
        if (row, col) != (0, 0):
            errors.append("pixel_coordinate_roundtrip")
    except Exception:
        errors.append("pixel_coordinate_roundtrip")
    return errors


def grid_errors(reference, source, tolerance_pixels, compare_grid=True):
    """Confronta CRS e, se richiesto, griglia/estensione col raster di riferimento."""
    errors = []
    if reference.crs is None or source.crs is None:
        return ["missing_crs"], None

    if reference.crs != source.crs:
        errors.append("crs")

    if not compare_grid:
        return errors, None

    if (reference.width, reference.height) != (source.width, source.height):
        errors.append("dimensions")

    resolution = min(abs(reference.transform.a), abs(reference.transform.e))
    if resolution == 0:
        errors.append("invalid_reference_resolution")
        return errors, None

    transform_error = max(
        abs(reference.transform.a - source.transform.a),
        abs(reference.transform.e - source.transform.e),
        abs(reference.transform.c - source.transform.c),
        abs(reference.transform.f - source.transform.f),
    ) / resolution

    try:
        source_bounds = transform_bounds(
            source.crs, reference.crs, *source.bounds
        )
        bounds_error = max(
            abs(left - right)
            for left, right in zip(reference.bounds, source_bounds)
        ) / resolution
    except Exception:
        errors.append("bounds_transform_failed")
        bounds_error = float("inf")

    if transform_error > tolerance_pixels:
        errors.append("transform")
    if bounds_error > tolerance_pixels:
        errors.append("bounds")
    return errors, max(transform_error, bounds_error)


def inspect_raster(
    reference_path,
    source_path,
    name,
    tolerance_pixels,
    compare_grid=True,
    coverage=False,
):
    row = base_row(name, source_path)
    if not source_path.exists():
        row["errors"] = ["missing"]
        return row

    try:
        with rasterio.open(source_path) as source:
            if reference_path.exists():
                with rasterio.open(reference_path) as reference:
                    errors, alignment_error = grid_errors(
                        reference, source, tolerance_pixels, compare_grid=compare_grid
                    )
            else:
                # Senza pre.tif il confronto non è possibile, ma il file
                # viene comunque letto e controllato da solo.
                errors, alignment_error = ["missing_reference"], None
                if source.crs is None:
                    errors.append("missing_crs")
            errors.extend(axis_errors(source))
            row.update(
                crs=str(source.crs) if source.crs else None,
                width=source.width,
                height=source.height,
                resolution_x=source.transform.a,
                resolution_y=source.transform.e,
                alignment_error_pixels=alignment_error,
                errors=errors,
            )
            if coverage:
                data = source.read(1, masked=True)
                valid = (~np.ma.getmaskarray(data)) & (np.asarray(data) > 0)
                row["coverage"] = float(valid.mean()) if valid.size else None
    except Exception as exc:
        row["errors"] = [f"read_error:{type(exc).__name__}:{exc}"]
    return row


def inspect_reference(path):
    row = base_row("planet_pre", path)
    if not path.exists():
        row["errors"] = ["missing"]
        return row
    try:
        with rasterio.open(path) as ds:
            row.update(
                crs=str(ds.crs) if ds.crs else None,
                width=ds.width,
                height=ds.height,
                resolution_x=ds.transform.a,
                resolution_y=ds.transform.e,
                errors=(["missing_crs"] if ds.crs is None else axis_errors(ds)),
            )
    except Exception as exc:
        row["errors"] = [f"read_error:{type(exc).__name__}:{exc}"]
    return row


def make_plot(pre_path, s2_path, mask_path, output_path):
    """Crea un plot diagnostico; riallinea solo la maschera per disegnare il contorno."""
    import matplotlib.pyplot as plt
    from rasterio.warp import reproject, Resampling

    with rasterio.open(pre_path) as src:
        planet = src.read([1, 2, 3]).astype(np.float32)
        transform, crs = src.transform, src.crs
        height, width = src.height, src.width
    with rasterio.open(s2_path) as src:
        if src.count < 3:
            logging.warning("Plot saltato: meno di 3 bande in %s", s2_path)
            return
        sentinel = src.read([3, 2, 1]).astype(np.float32)
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
        aligned_mask = np.zeros((height, width), dtype=np.uint8)
        reproject(
            source=mask,
            destination=aligned_mask,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.nearest,
        )

    def rgb(array):
        low = np.percentile(array, 2, axis=(1, 2), keepdims=True)
        high = np.percentile(array, 99, axis=(1, 2), keepdims=True)
        scaled = (array - low) / np.maximum(high - low, 1)
        return np.moveaxis(np.clip(scaled, 0, 1), 0, -1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    axes[0].imshow(rgb(planet))
    axes[0].contour(aligned_mask > 0, colors="yellow", linewidths=0.6)
    axes[0].set_title("PlanetScope pre + mask")
    axes[0].axis("off")
    axes[1].imshow(rgb(sentinel))
    axes[1].set_title("Sentinel-2 RGB")
    axes[1].axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def get_date_dirs(patch, requested_dates=None):
    s2_root = patch / "s2"
    if not s2_root.is_dir():
        return []
    date_dirs = sorted(p for p in s2_root.iterdir() if p.is_dir())
    if requested_dates:
        requested = set(requested_dates)
        date_dirs = [p for p in date_dirs if p.name in requested]
    return date_dirs


# ---------------------------------------------------------------------------
# Controllo di una patch
# ---------------------------------------------------------------------------

def inspect_patch(
    patch,
    sensor,
    tolerance_pixels,
    event_date=None,
    requested_dates=None,
    plots_dir=None,
    sentinel_strict_grid=False,
    min_coverage=0.01,
    require_planet_dates=False,
):
    """Restituisce (righe raster, riepilogo temporale della patch)."""
    event = patch.parent.name
    reference = patch / "pre.tif"
    rows = []
    planet_rows = {}
    s2_by_class = {"pre": [], "post": [], "same_day": [], "unknown": []}

    if sensor in ("planet", "both"):
        pre_row = inspect_reference(reference)
        post_row = inspect_raster(
            reference, patch / "post.tif", "planet_post", tolerance_pixels, compare_grid=True
        )
        mask_row = inspect_raster(
            reference, patch / "mask.tif", "mask", tolerance_pixels, compare_grid=True
        )
        for expected, row in (("pre", pre_row), ("post", post_row)):
            path = Path(row["path"])
            if path.exists():
                acquisition, source = read_planet_date(path)
                row.update(
                    acquisition_date=acquisition.isoformat() if acquisition else None,
                    date_source=source,
                    temporal_class=classify_date(acquisition, event_date),
                )
            planet_rows[expected] = row
        rows.extend((pre_row, post_row, mask_row))

    if sensor in ("sentinel", "both"):
        all_date_dirs = get_date_dirs(patch)
        date_dirs = get_date_dirs(patch, requested_dates)

        if not all_date_dirs:
            row = base_row("sentinel_directory", patch / "s2")
            row["errors"] = ["missing_or_empty_s2_directory"]
            rows.append(row)
        if requested_dates:
            found_names = {p.name for p in all_date_dirs}
            for requested_date in requested_dates:
                if requested_date not in found_names:
                    row = base_row("sentinel_date", patch / "s2" / requested_date, requested_date)
                    row["errors"] = ["date_not_found"]
                    rows.append(row)

        for date_dir in date_dirs:
            acquisition = parse_date(date_dir.name)
            temporal_class = classify_date(acquisition, event_date)
            date_rows = {}
            for name, filename in (
                ("s2_10m", "s2_10m.tif"),
                ("s2_20m", "s2_20m.tif"),
                ("s2_valid", "s2_valid.tif"),
            ):
                result = inspect_raster(
                    reference,
                    date_dir / filename,
                    name,
                    tolerance_pixels,
                    compare_grid=sentinel_strict_grid,
                    coverage=(name == "s2_valid"),
                )
                result.update(
                    date=date_dir.name,
                    acquisition_date=acquisition.isoformat() if acquisition else None,
                    date_source="s2_folder_name",
                    temporal_class=temporal_class,
                )
                if acquisition is None:
                    result["warnings"].append("unparseable_date")
                date_rows[name] = result
                rows.append(result)

            # Una data conta se l'immagine e la maschera di validità sono
            # leggibili e la copertura valida è sufficiente.
            valid_row = date_rows["s2_valid"]
            usable = (
                not date_rows["s2_10m"]["errors"]
                and not valid_row["errors"]
                and valid_row["coverage"] is not None
                and valid_row["coverage"] >= min_coverage
            )
            if usable:
                s2_by_class[temporal_class].append(date_dir.name)
            elif not date_rows["s2_10m"]["errors"] and not valid_row["errors"]:
                valid_row["warnings"].append("low_coverage")

            if (
                plots_dir
                and (date_dir / "s2_10m.tif").exists()
                and (patch / "mask.tif").exists()
                and reference.exists()
            ):
                try:
                    make_plot(
                        reference,
                        date_dir / "s2_10m.tif",
                        patch / "mask.tif",
                        plots_dir / f"{event}_{patch.name}_{date_dir.name}.png",
                    )
                except Exception as exc:
                    logging.warning("Plot non creato per %s, data %s: %s", patch, date_dir.name, exc)

    # Riepilogo temporale della patch.
    summary = {
        "event": event,
        "patch_id": patch.name,
        "event_date": event_date.isoformat() if event_date else None,
        "s2_pre_dates": s2_by_class["pre"],
        "s2_post_dates": s2_by_class["post"],
        "s2_same_day_dates": s2_by_class["same_day"],
        "s2_undated": s2_by_class["unknown"] if event_date else [],
        "planet_pre_date": planet_rows.get("pre", {}).get("acquisition_date"),
        "planet_post_date": planet_rows.get("post", {}).get("acquisition_date"),
        "errors": [],
        "warnings": [],
    }
    if event_date is None:
        summary["errors"].append("event_date_missing")
    else:
        if sensor in ("sentinel", "both"):
            if not s2_by_class["pre"]:
                summary["errors"].append("no_usable_s2_pre")
            if not s2_by_class["post"]:
                summary["errors"].append("no_usable_s2_post")
            if s2_by_class["same_day"]:
                summary["warnings"].append("s2_same_day_as_event")
        for expected, row in planet_rows.items():
            if not Path(row["path"]).exists():
                summary["errors"].append(f"planet_{expected}_missing")
                continue
            temporal_class = row["temporal_class"]
            if temporal_class == "unknown":
                target = summary["errors"] if require_planet_dates else summary["warnings"]
                target.append(f"planet_{expected}_date_unknown")
            elif temporal_class != expected:
                when = "before" if expected == "pre" else "after"
                summary["errors"].append(f"planet_{expected}_not_{when}_event")

    for result in rows:
        result.update(event=event, patch_id=patch.name, sensor=sensor)
        result["passed"] = not result["errors"]
        result["errors"] = ";".join(result["errors"])
        result["warnings"] = ";".join(result["warnings"])

    summary["passed"] = not summary["errors"]
    for key in ("s2_pre_dates", "s2_post_dates", "s2_same_day_dates", "s2_undated",
                "errors", "warnings"):
        summary[key] = ";".join(summary[key])
    return rows, summary


def summarize_events(patch_summaries):
    by_event = {}
    for item in patch_summaries:
        by_event.setdefault(item["event"], []).append(item)

    def split(value):
        return [v for v in value.split(";") if v]

    events = []
    for event, items in sorted(by_event.items()):
        failing = [item for item in items if not item["passed"]]
        errors = sorted({e for item in items for e in split(item["errors"])})
        events.append({
            "event": event,
            "event_date": items[0]["event_date"],
            "patches_checked": len(items),
            "patches_passed": len(items) - len(failing),
            "patches_with_s2_pre": sum(bool(item["s2_pre_dates"]) for item in items),
            "patches_with_s2_post": sum(bool(item["s2_post_dates"]) for item in items),
            "s2_pre_dates": ";".join(sorted({d for i in items for d in split(i["s2_pre_dates"])})),
            "s2_post_dates": ";".join(sorted({d for i in items for d in split(i["s2_post_dates"])})),
            "errors": ";".join(errors),
            "failing_patches": ";".join(item["patch_id"] for item in failing),
            "passed": not failing,
        })
    return events


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def safe_date_filename(value):
    # Evita caratteri non validi nei nomi file, mantenendo la data leggibile.
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))


def write_report(rows, output_dir, filename_prefix, kind="geospatial"):
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{filename_prefix}_{kind}_report.csv"
    json_path = output_dir / f"{filename_prefix}_{kind}_report.json"
    json_path.write_text(
        json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    fields = sorted({key for item in rows for key in item})
    if not fields:
        fields = ["event", "patch_id", "passed", "errors"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(not item.get("passed", False) for item in rows)
    logging.info(
        "%s: salvati %s e %s (%d record, %d con errori)",
        filename_prefix, csv_path, json_path, len(rows), failures
    )
    return failures


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    event_dates = load_event_dates(args.event_date, args.event_dates)

    all_patches = discover_patches(args.patches_root, args.events)
    patches = sample_patches(all_patches, args.max_samples, args.seed)
    if not patches:
        raise RuntimeError(f"Nessuna cartella patch numerica trovata in {args.patches_root}")
    if len(patches) < len(all_patches):
        logging.warning(
            "Controllo su un campione di %d patch su %d: il controllo pre/post per "
            "evento vale solo per le patch campionate.", len(patches), len(all_patches)
        )

    found_events = {p.parent.name for p in patches}
    missing_dates = sorted(found_events - set(event_dates))
    if missing_dates:
        logging.warning(
            "Data evento non fornita per: %s. Per questi eventi pre/post non può "
            "essere verificato.", ", ".join(missing_dates)
        )

    plots_dir = args.output_dir / "plots" if args.plots else None
    rows, patch_summaries = [], []
    for patch in patches:
        patch_rows, summary = inspect_patch(
            patch,
            sensor=args.sensor,
            tolerance_pixels=args.tolerance_pixels,
            event_date=event_dates.get(patch.parent.name),
            requested_dates=args.dates,
            plots_dir=plots_dir,
            sentinel_strict_grid=args.sentinel_strict_grid,
            min_coverage=args.min_coverage,
            require_planet_dates=args.require_planet_dates,
        )
        rows.extend(patch_rows)
        patch_summaries.append(summary)

    geo_failures = 0

    # Report PlanetScope separato.
    if args.sensor in ("planet", "both"):
        planet_rows = [
            r for r in rows if r["raster"] in ("planet_pre", "planet_post", "mask")
        ]
        geo_failures += write_report(planet_rows, args.output_dir, "planet")

    # Report Sentinel generale e un report dedicato a ciascuna data.
    if args.sensor in ("sentinel", "both"):
        sentinel_rows = [
            r for r in rows
            if r["raster"].startswith("s2_")
            or r["raster"] in ("sentinel_directory", "sentinel_date")
        ]
        geo_failures += write_report(sentinel_rows, args.output_dir, "sentinel")

        # I report per data sono sottoinsiemi del report generale: i loro
        # errori non vengono sommati di nuovo al totale.
        dates_found = sorted({
            str(r["date"]) for r in sentinel_rows if r.get("date") not in (None, "")
        })
        for date_name in dates_found:
            date_rows = [r for r in sentinel_rows if str(r.get("date")) == date_name]
            write_report(date_rows, args.output_dir, f"sentinel_{safe_date_filename(date_name)}")

    # Report temporali: per patch e per evento.
    write_report(patch_summaries, args.output_dir, "patches", kind="temporal")
    event_rows = summarize_events(patch_summaries)
    event_failures = write_report(event_rows, args.output_dir, "events", kind="temporal")
    for item in event_rows:
        level = logging.INFO if item["passed"] else logging.WARNING
        logging.log(
            level,
            "Evento %s (data %s): %d/%d patch con pre e post validi%s",
            item["event"], item["event_date"] or "mancante",
            item["patches_passed"], item["patches_checked"],
            "" if item["passed"] else f" - errori: {item['errors']}",
        )

    if geo_failures or event_failures:
        raise SystemExit(
            f"Controlli completati: {geo_failures} record geospaziali con errori, "
            f"{event_failures} eventi senza pre/post validi."
        )
    logging.info("Controlli completati per %d patch.", len(patches))


if __name__ == "__main__":
    main()