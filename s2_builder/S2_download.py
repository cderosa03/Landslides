import json
import logging
import os
import re
import requests
import zipfile

from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


# Logging Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DOWNLOAD_DIR = Path(os.getenv("S2_IMAGES_PATH", PROJECT_ROOT / "Sentinel" / "images"))
CREDENTIALS_PATH = Path(
    os.getenv("S2_CREDENTIALS_PATH", Path(__file__).with_name("credentials.json"))
)
PRODUCT_LEVEL = "MSIL2A"

ACCESS_TOKEN = None
REFRESH_TOKEN = None


# Load login credentials from JSON
def load_credentials():
    try:
        with open(CREDENTIALS_PATH) as f:
            return json.load(f)
    except Exception as e:
        logging.critical(f"Failed to load login credentials from {CREDENTIALS_PATH}: {e}")
        raise


INVENTORIES = [
    {
        "name": "Lombok2018",
        "tiles": [{"id": "T50LLR", "ron": "R003"}, {"id": "T50LMR", "ron": "R003"}],
        "start_date": "2018-06-01T00:00:00.000Z",
        "end_date": "2018-07-01T00:00:00.000Z",
    },
    {
        "name": "Philippines2019",
        "tiles": [{"id": "T51NYH", "ron": "R060"}],
        "start_date": "2019-09-01T00:00:00.000Z",
        "end_date": "2020-03-01T00:00:00.000Z",
    },
    {
        "name": "Michoacan2022",
        "tiles": [
            {"id": "T13QFB", "ron": "R012"},
            {"id": "T13QFA", "ron": "R012"},
            {"id": "T13QGB", "ron": "R112"},
            {"id": "T13QGA", "ron": "R112"},
        ],
        "start_date": "2022-08-01T00:00:00.000Z",
        "end_date": "2022-11-01T00:00:00.000Z",
    },
    {
        "name": "EmiliaRomagna2023",
        "tiles": [
            {"id": "T32TPQ", "ron": "R022"},
            {"id": "T32TPP", "ron": "R022"},
            {"id": "T32TQQ", "ron": "R022"},
            {"id": "T32TQP", "ron": "R022"},
        ],
        "start_date": "2023-04-01T00:00:00.000Z",
        "end_date": "2023-07-01T00:00:00.000Z",
    },
]

def get_access_token(username: str, password: str):
    """Request a new access_token and refresh_token."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "password",
        "username": username,
        "password": password,
        "client_id": "cdse-public",
    }

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        token_info = response.json()
        logging.info("Access token retrieved successfully.")
        return token_info.get("access_token"), token_info.get("refresh_token")
    except requests.RequestException as e:
        logging.error(f"Failed to retrieve access token: {e}")
        return None, None


def regenerate_access_token(refresh_token: str):
    """Regenerate the access_token using the refresh_token."""
    url = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
    headers = {"Content-Type": "application/x-www-form-urlencoded"}
    data = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": "cdse-public",
    }

    try:
        response = requests.post(url, headers=headers, data=data)
        response.raise_for_status()
        token_info = response.json()
        logging.info("Access token regenerated successfully.")
        return token_info.get("access_token")
    except requests.RequestException as e:
        logging.error(f"Error regenerating access token: {e}")
        return None


def handle_token_expiry(credentials):
    """Handle token expiry by refreshing or requesting a new access token."""
    global ACCESS_TOKEN, REFRESH_TOKEN
    ACCESS_TOKEN = regenerate_access_token(REFRESH_TOKEN)

    if not ACCESS_TOKEN:
        logging.warning("Refresh token expired. Attempting re-authentication...")
        username, password = credentials["username"], credentials["password"]
        ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(username, password)
        if not ACCESS_TOKEN:
            logging.critical("Re-authentication failed. Check your credentials.")
            raise Exception("Authentication failed.")


def fetch_products(params):
    """Fetch filtered products and return a list of the most recent baseline products per date."""
    products_by_date = defaultdict(list)
    baseline_pattern = r"_N(\d{4})_"
    base_url = url = "https://catalogue.dataspace.copernicus.eu/odata/v1/Products"

    logging.info("Fetching products from API...")

    while url:
        try:
            response = requests.get(url, params=params if url == base_url else None)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Error fetching products: {e}")
            break

        data = response.json()
        for product in data["value"]:
            name = product["Name"]
            date = product["ContentDate"]["Start"][:10]

            match = re.search(baseline_pattern, name)
            if match:
                baseline_number = int(match.group(1))
                content_length = product["ContentLength"]
                products_by_date[date].append(
                    (baseline_number, content_length, product)
                )

        url = data.get("@odata.nextLink", None)

    # Process and log results
    product_list = []
    for date, products in products_by_date.items():
        # Get the product with the highest baseline product and, in the case of a tie, the highest content length.
        max_baseline_product = max(products, key=lambda x: (x[0], x[1]))[2]
        product_list.append(max_baseline_product)
        logging.debug(f"Date: {date}, Product: {max_baseline_product['Name']}")

    logging.info(f"Total products fetched: {len(product_list)}")

    return product_list


def download_product(product, inventory_name, product_level, credentials):
    """Download and unzip a specific product."""
    global ACCESS_TOKEN
    product_id = product["Id"]
    product_name = product["Name"]
    file_name = product_name.split(".")[0] + ".zip"
    mgrs = product_name.split("_")[-2]
    file_path = Path(DOWNLOAD_DIR, inventory_name, mgrs, product_level, file_name)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    if file_path.with_suffix(".SAFE").exists():
        logging.info(f"{file_path.with_suffix('.SAFE').name} already extracted. Skipping.")
        return

    if file_path.exists():
        logging.info(f"{file_name} already exists. Will extract next.")
        unzip_product(file_path)
        return

    url = f"https://download.dataspace.copernicus.eu/odata/v1/Products({product_id})/$value"
    attempt = 0
    while attempt < 3:
        headers = {"Authorization": f"Bearer {ACCESS_TOKEN}"}
        session = requests.Session()
        session.headers.update(headers)

        try:
            response = session.get(url, stream=True)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.warning(f"Download failed for {product_name}: {e}")
            handle_token_expiry(credentials)
            attempt += 1
            if attempt >= 3:
                logging.error(f"Too many failed attempts for {product_name}.")
                raise
            continue

        total_size = int(response.headers.get("content-length", 0))
        with open(file_path, "wb") as file:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=f"Downloading {product_name}",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
                        pbar.update(len(chunk))
        logging.info(f"Download completed for {product_name}.")
        break

    unzip_product(file_path)


def unzip_product(zip_path: Path):
    """Unzip a single .zip product if not already extracted, then delete the zip."""
    try:
        safe_folder = zip_path.with_suffix(".SAFE")
        if safe_folder.exists():
            logging.info(f"{safe_folder.name} already exists. Skipping extraction.")
            return

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            logging.info(f"Extracting {zip_path.name}...")
            zip_ref.extractall(zip_path.parent)

        zip_path.unlink()
        logging.info(f"Deleted {zip_path.name} after extraction.")
    except Exception as e:
        logging.error(f"Failed to extract {zip_path.name}: {e}")


def main():
    credentials = None
    for inventory in INVENTORIES:
        inventory_name = inventory["name"]
        start_date = inventory["start_date"]
        end_date = inventory["end_date"]
        tiles = inventory["tiles"]

        product_level = PRODUCT_LEVEL
        logging.info(f"Starting downloads for product level: {product_level}")
            
        for tile in tiles:
            tile_id = tile["id"]
            tile_ron = tile["ron"]
            logging.info(f"Processing tile: {tile_id}")
        
            params = {
                "$filter": f"Collection/Name eq 'SENTINEL-2' and "
                f"contains(Name, '{tile_id}') and "
                f"contains(Name, '{tile_ron}') and "
                f"contains(Name, '{product_level}') and "
                f"ContentDate/Start gt {start_date} and "
                f"ContentDate/Start lt {end_date}",
                "$orderby": "ContentDate/Start asc",
            }

            try:
                products = fetch_products(params)
                logging.info(f"Found {len(products)} products for tile {tile_id}.")

                # Ensure tokens are loaded only once an actual download is needed.
                if products and not ACCESS_TOKEN:
                    credentials = credentials or load_credentials()
                    username, password = credentials["username"], credentials["password"]
                    ACCESS_TOKEN, REFRESH_TOKEN = get_access_token(username, password)

                for i, product in enumerate(products):
                    logging.info(f"Downloading product {i + 1}/{len(products)}")
                    download_product(product, inventory_name, product_level, credentials)
            except Exception as e:
                logging.error(f"Error processing tile {tile_id}: {e}")

    logging.info("All downloads completed.")


if __name__ == "__main__":
    main()
