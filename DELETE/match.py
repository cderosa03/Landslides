import shutil
import logging
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

SOURCE_BASE = Path(
    "/home/jovyan/nfs/tesista3/landslide-detection/Landslides/PlanetScope/patches"
)
TARGET_BASE = Path(
    "/home/jovyan/nfs/tesista3/landslide-detection/Landslides/matched"
)

EVENT_RANGES = {
    "Lombok2018": {
        "pre": ("2018-06-01", "2018-08-04"),
        "post": ("2018-08-06", "2018-09-30"),
    },
    "Philippines2019": {
        "pre": ("2019-06-01", "2019-07-24"),
        "post": ("2019-07-26", "2019-09-30"),
    },
    "Michoacan2022": {
        "pre": ("2022-08-01", "2022-09-18"),
        "post": ("2022-09-20", "2022-10-15"),
    },
    "EmiliaRomagna2023": {
        "pre": ("2023-04-01", "2023-05-15"),
        "post": ("2023-05-17", "2023-06-15"),
    },
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# --------------------------------------------------
# UTILS
# --------------------------------------------------

def parse_patch_date(date_str: str) -> datetime:
    """Parse YYYYMMDD folder name."""
    return datetime.strptime(date_str, "%Y%m%d")


def parse_cfg_date(date_str: str) -> datetime:
    """Parse YYYY-MM-DD config date."""
    return datetime.strptime(date_str, "%Y-%m-%d")


def copy_files(src_date_dir: Path, dst_dir: Path):
    """Copy s2_10m, s2_20m, and optionally y.tif and dtm.tif."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    files_to_copy = ["s2_10m.tif", "s2_20m.tif", "y.tif", "dtm.tif"]

    for fname in files_to_copy:
        src = src_date_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
# --------------------------------------------------
# MAIN
# --------------------------------------------------

for event, ranges in EVENT_RANGES.items():
    logging.info(f"Processing event: {event}")

    event_src = SOURCE_BASE / event
    event_dst = TARGET_BASE / event 

    if not event_src.exists():
        logging.warning(f"Event directory not found: {event_src}")
        continue

    pre_start  = parse_cfg_date(ranges["pre"][0])
    pre_end    = parse_cfg_date(ranges["pre"][1])
    post_start = parse_cfg_date(ranges["post"][0])
    post_end   = parse_cfg_date(ranges["post"][1])

    for patch_dir in sorted(event_src.iterdir()):
        if not patch_dir.is_dir() or not patch_dir.name.isdigit():
            continue

        s2_dir = patch_dir / "s2"
        if not s2_dir.exists():
            continue

        all_dates = []
        for d in s2_dir.iterdir():
            if d.is_dir():
                try:
                    all_dates.append((parse_patch_date(d.name), d))
                except ValueError:
                    continue

        if not all_dates:
            continue

        # ---- FILTER PRE/POST ----
        pre_dates = [
            (date, d) for date, d in all_dates if pre_start <= date <= pre_end
        ]
        post_dates = [
            (date, d) for date, d in all_dates if post_start <= date <= post_end
        ]

        if not pre_dates and not post_dates:
            continue

        out_patch_dir = event_dst / patch_dir.name

        # ---- COPY ALL PRE ----
        for date, date_dir in sorted(pre_dates, key=lambda x: x[0]):
            try:
                copy_files(
                    date_dir,
                    out_patch_dir / "pre" / date.strftime("%Y%m%d")
                )
            except Exception as e:
                logging.warning(
                    f"{event} | patch {patch_dir.name} | pre {date.strftime('%Y%m%d')} skipped: {e}"
                )

        # ---- COPY ALL POST ----
        for date, date_dir in sorted(post_dates, key=lambda x: x[0]):
            try:
                copy_files(
                    date_dir,
                    out_patch_dir / "post" / date.strftime("%Y%m%d")
                )
            except Exception as e:
                logging.warning(
                    f"{event} | patch {patch_dir.name} | post {date.strftime('%Y%m%d')} skipped: {e}"
                )

        logging.info(
            f"{event} | patch {patch_dir.name} | "
            f"pre={len(pre_dates)} post={len(post_dates)}"
        )

logging.info("Matching (ALL pre/post + y.tif/dtm.tif) completed successfully.")
