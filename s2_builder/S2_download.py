#!/usr/bin/env python3
"""Inventory and safely download missing Copernicus Sentinel-2 L2A products.

Default mode is DRY RUN. Use --download to transfer data. Existing .SAFE
folders are never overwritten; incomplete folders are reported for manual review.
"""
import argparse
import csv
import json
import logging
import os
import re
import shutil
import sys
import time
import zipfile
from datetime import date
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = Path(os.getenv("S2_IMAGES_PATH", PROJECT_ROOT / "Landslides" / "Sentinel" / "images"))
CREDENTIALS_PATH = Path(os.getenv("S2_CREDENTIALS_PATH", Path(__file__).with_name("credentials.json")))
CATALOG = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1/Products({})/$value"
LEVEL = "MSIL2A"

INVENTORIES = {
    "Lombok2018": {"tiles": ["T50LLR", "T50LMR"], "start": "2018-06-01", "end": "2018-10-01", "cutoff": "2018-08-05"},
    "Philippines2019": {"tiles": ["T51NYH"], "start": "2019-09-01", "end": "2020-03-01", "cutoff": "2019-10-16"},
    "Michoacan2022": {"tiles": ["T13QFB", "T13QFA", "T13QGB", "T13QGA"], "start": "2022-08-01", "end": "2022-11-01", "cutoff": "2022-09-19"},
    "EmiliaRomagna2023": {"tiles": ["T32TPQ", "T32TPP", "T32TQQ", "T32TQP"], "start": "2023-04-01", "end": "2023-07-01", "cutoff": "2023-05-16"},
}
# Name: S2A_MSIL2A_YYYYMMDDTHHMMSS_N0500_R003_T50LLR_YYYYMMDDTHHMMSS.SAFE
NAME_RE = re.compile(r"^(S2[AB])_MSIL2A_(\d{8}T\d{6})_N(\d{4})_(R\d{3})_(T\d{2}[A-Z]{3})_(\d{8}T\d{6})\.SAFE$")
LOG = logging.getLogger("s2_download")


def args_parse():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events", nargs="+", choices=list(INVENTORIES), default=["Lombok2018", "Philippines2019"])
    p.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    p.add_argument("--report", type=Path, default=Path("s2_download_inventory.csv"))
    p.add_argument("--download", action="store_true", help="Actually download (otherwise dry run)")
    p.add_argument("--post-only", action="store_true", help="Only products acquired strictly after event date")
    p.add_argument("--max-downloads", type=int, default=0, help="Maximum downloads (0 = unlimited)")
    p.add_argument("--min-free-gb", type=float, default=500, help="Safety reserve on destination filesystem")
    p.add_argument("--timeout", type=int, default=120)
    return p.parse_args()


def request_with_retry(session, method, url, *, attempts=5, **kwargs):
    timeout = kwargs.pop("_timeout", 120)
    for attempt in range(attempts):
        try:
            resp = session.request(method, url, timeout=timeout, **kwargs)
            if resp.status_code in (429, 500, 502, 503, 504):
                delay = min(60, 2 ** attempt * 3)
                resp.close()
                if attempt + 1 < attempts:
                    time.sleep(delay)
                    continue
            resp.raise_for_status()
            return resp
        except (requests.Timeout, requests.ConnectionError):
            if attempt + 1 == attempts:
                raise
            time.sleep(min(60, 2 ** attempt * 3))
    raise RuntimeError("Request retries exhausted")


def catalogue_for_tile(session, tile, inventory, timeout):
    # Do NOT filter relative orbit: e.g. Philippines T51NYH also has R017.
    expr = ("Collection/Name eq 'SENTINEL-2' and "
            "contains(Name, 'MSIL2A') and "
            f"contains(Name, '{tile}') and "
            f"ContentDate/Start gt {inventory['start']}T00:00:00.000Z and "
            f"ContentDate/Start lt {inventory['end']}T00:00:00.000Z")
    params = {"$filter": expr, "$select": "Id,Name,ContentLength,ContentDate", "$top": 100}
    url = CATALOG
    products = []
    seen_urls = set()
    while url:
        if url in seen_urls:
            raise RuntimeError("Catalog pagination loop")
        seen_urls.add(url)
        with request_with_retry(session, "GET", url, params=params if url == CATALOG else None, _timeout=timeout) as response:
            obj = response.json()
        for product in obj.get("value", []):
            parsed = NAME_RE.fullmatch(product.get("Name", ""))
            if parsed and parsed.group(5) == tile:
                products.append(product)
        url = obj.get("@odata.nextLink")
    return products


def select_products(products):
    # Select newest processing baseline, then newest generation, per satellite/acquisition/orbit/tile.
    # This avoids arbitrary selection by file size and preserves different acquisition times.
    chosen = {}
    for p in products:
        m = NAME_RE.fullmatch(p["Name"])
        satellite, acquisition, baseline, orbit, tile, generated = m.groups()
        key = (satellite, acquisition, orbit, tile)
        score = (int(baseline), generated, int(p.get("ContentLength") or 0))
        if key not in chosen or score > chosen[key][0]:
            chosen[key] = (score, p)
    return sorted((x[1] for x in chosen.values()), key=lambda p: p["Name"])


def safe_folder_ok(path):
    if not path.is_dir():
        return False
    manifest = path / "manifest.safe"
    if not manifest.is_file():
        manifest = path / "MTD_MSIL2A.xml"
    return manifest.is_file() and any(path.rglob("*.jp2"))


def zip_looks_ok(path):
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as z:
            return z.testzip() is None and bool(z.namelist())
    except (zipfile.BadZipFile, OSError):
        return False


def file_status(destination):
    folder = destination.with_suffix(".SAFE")
    if folder.exists():
        return "present" if safe_folder_ok(folder) else "incomplete_safe"
    if destination.exists():
        return "zip_complete" if zip_looks_ok(destination) else "incomplete_zip"
    return "missing"


def get_token(session, credentials, timeout):
    response = request_with_retry(
        session, "POST", TOKEN_URL,
        data={"grant_type": "password", "client_id": "cdse-public", "username": credentials["username"], "password": credentials["password"]},
        _timeout=timeout,
    )
    try:
        return response.json()["access_token"]
    finally:
        response.close()


def load_credentials():
    with CREDENTIALS_PATH.open(encoding="utf-8") as f:
        credentials = json.load(f)
    if not credentials.get("username") or not credentials.get("password"):
        raise ValueError("credentials.json must contain username and password")
    return credentials


def extract_zip_atomic(path, destination_dir):
    """Stage ZIP extraction beside destination; refuse unsafe entries and existing targets."""
    target = path.with_suffix(".SAFE")
    if target.exists():
        raise FileExistsError(f"Refusing to replace existing {target}")
    stage = destination_dir / (".unpacking_" + target.name)
    if stage.exists():
        raise FileExistsError(f"Old extraction stage exists: {stage}")
    stage.mkdir()
    try:
        with zipfile.ZipFile(path) as z:
            infos = z.infolist()
            for member in infos:
                parts = Path(member.filename).parts
                if (not parts or any(x in ("..", "") for x in parts)
                        or Path(member.filename).is_absolute() or parts[0] != target.name
                        or member.is_dir() is False and member.file_size < 0):
                    raise ValueError(f"Unsafe ZIP entry: {member.filename}")
                # Reject symlinks
                if (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("ZIP symlink not permitted")
            if z.testzip() is not None:
                raise ValueError("ZIP CRC check failed")
            z.extractall(stage)
        extracted = stage / target.name
        if not safe_folder_ok(extracted):
            raise ValueError("Extracted product lacks required SAFE structure or JP2 bands")
        extracted.rename(target)
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def perform_download(session, item, destination, credentials, args):
    status = file_status(destination)
    if status == "present":
        return "already_present"
    if status.startswith("incomplete"):
        return "manual_review_" + status
    destination.parent.mkdir(parents=True, exist_ok=True)
    if status == "zip_complete":
        extract_zip_atomic(destination, destination.parent)
        return "extracted_existing_zip"

    size = int(item.get("ContentLength") or 0)
    free = shutil.disk_usage(destination.parent).free
    reserve = int(args.min_free_gb * 1024**3)
    # Extraction may temporarily require 2x space. Refuse low-space cases.
    if free < reserve + 2 * size:
        return "insufficient_space"
    part = destination.with_suffix(".zip.part")
    if part.exists():
        return "manual_review_partial_zip"
    url = DOWNLOAD_URL.format(item["Id"])
    for attempt in range(2):
        token = get_token(session, credentials, args.timeout)
        try:
            with session.get(url, headers={"Authorization": f"Bearer {token}"}, stream=True, timeout=(30, args.timeout)) as response:
                if response.status_code == 401 and attempt == 0:
                    continue
                response.raise_for_status()
                with part.open("xb") as out:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            out.write(chunk)
            break
        except Exception:
            part.unlink(missing_ok=True)
            raise
    else:
        return "authentication_failed"
    if size and part.stat().st_size != size:
        return "download_size_mismatch_partial_kept"
    if not zip_looks_ok(part):
        return "download_crc_failed_partial_kept"
    part.rename(destination)
    extract_zip_atomic(destination, destination.parent)
    # Only remove ZIP once extraction was verified and installed.
    destination.unlink()
    return "downloaded_and_extracted"


def main():
    args = args_parse()
    if args.max_downloads < 0 or args.min_free_gb < 0:
        raise SystemExit("--max-downloads and --min-free-gb must be nonnegative")
    args.root = args.root.expanduser().resolve()
    session = requests.Session()
    rows = []
    seen_names = set()
    for event in args.events:
        config = INVENTORIES[event]
        cutoff = config["cutoff"].replace("-", "")
        for tile in config["tiles"]:
            LOG.info("Query %s / %s", event, tile)
            try:
                products = select_products(catalogue_for_tile(session, tile, config, args.timeout))
            except Exception as exc:
                LOG.error("Catalog error for %s/%s: %s", event, tile, exc)
                raise SystemExit(2) from exc
            for product in products:
                name = product["Name"]
                if (event, name) in seen_names:
                    continue
                seen_names.add((event, name))
                match = NAME_RE.fullmatch(name)
                acquired = match.group(2)[:8]
                period = "pre" if acquired < cutoff else "post" if acquired > cutoff else "same_day"
                if args.post_only and period != "post":
                    continue
                dest = args.root / event / tile / LEVEL / (name[:-5] + ".zip")
                row = {"event": event, "tile": tile, "acquired": acquired, "period": period,
                       "product": name, "product_id": product["Id"], "bytes": int(product.get("ContentLength") or 0),
                       "status": file_status(dest), "destination": str(dest)}
                rows.append(row)
    count = len(rows)
    needs = [r for r in rows if r["status"] != "present"]
    bytes_missing = sum(r["bytes"] for r in needs if r["status"] == "missing")
    print(f"Products selected: {count}; not already extracted: {len(needs)}; missing download bytes (estimate): {bytes_missing/1024**3:.2f} GiB")
    if args.download:
        credentials = None
        downloaded = 0
        for row in rows:
            if row["status"] == "present":
                continue
            if args.max_downloads and downloaded >= args.max_downloads:
                row["result"] = "download_limit_reached"
                continue
            product = {"Id": row["product_id"], "ContentLength": row["bytes"]}
            try:
                if row["status"] == "missing":
                    if credentials is None:
                        credentials = load_credentials()
                result = perform_download(session, product, Path(row["destination"]), credentials, args)
                row["result"] = result
                LOG.info("%s: %s", row["product"], result)
                if result == "downloaded_and_extracted":
                    downloaded += 1
            except Exception as exc:
                row["result"] = f"error:{type(exc).__name__}:{exc}"
                LOG.error("Failure for %s: %s", row["product"], exc)
    else:
        LOG.info("DRY RUN: no product downloads or changes to existing data")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["event", "tile", "acquired", "period", "product", "product_id", "bytes", "status", "destination", "result"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved inventory: {args.report}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    main()
