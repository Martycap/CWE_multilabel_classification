import csv
import gzip
import json
import shutil
from datetime import datetime
from pathlib import Path

import requests


START_YEAR = 2002
END_YEAR = datetime.now().year

BASE_URL = "https://nvd.nist.gov/feeds/json/cve/2.0"

ZIP_DIR = Path("data/raw/zip")
JSON_DIR = Path("data/raw/json")
CSV_FILE = Path("data/cve_dataset.csv")


def download_json_files():
    """Download the compressed NVD JSON files, one for each year."""

    ZIP_DIR.mkdir(parents=True, exist_ok=True)

    for year in range(START_YEAR, END_YEAR + 1):
        url = f"{BASE_URL}/nvdcve-2.0-{year}.json.gz"
        zip_file = ZIP_DIR / f"nvdcve-2.0-{year}.json.gz"

        if zip_file.exists():
            print(f"{zip_file.name} already exists.")
            continue

        print(f"Downloading CVEs for {year}...")

        response = requests.get(url, timeout=120)
        response.raise_for_status()

        zip_file.write_bytes(response.content)


def unzip_json_files():
    """Decompress all downloaded .json.gz files."""

    JSON_DIR.mkdir(parents=True, exist_ok=True)

    for zip_file in sorted(ZIP_DIR.glob("*.json.gz")):
        json_file = JSON_DIR / zip_file.name.replace(".gz", "")

        if json_file.exists():
            print(f"{json_file.name} already exists.")
            continue

        print(f"Decompressing {zip_file.name}...")

        with gzip.open(zip_file, "rb") as compressed_file:
            with open(json_file, "wb") as output_file:
                shutil.copyfileobj(compressed_file, output_file)


def get_description(cve):
    """Return the English description of the CVE."""

    for description in cve.get("descriptions", []):
        if description.get("lang") == "en":
            return description.get("value", "")

    return ""


def get_unique_cwes(cve):
    """Extract all CWE identifiers associated with the CVE and remove duplicates."""

    cwe_set = set()

    for weakness in cve.get("weaknesses", []):
        for description in weakness.get("description", []):
            cwe_id = description.get("value", "").strip()

            if cwe_id:
                cwe_set.add(cwe_id)

    return sorted(cwe_set)


def create_csv():
    """Create the CSV file containing the CVE ID, description and CWE list."""

    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    seen_cve_ids = set()
    saved_rows = 0

    with open(CSV_FILE, "w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["cve_id", "description", "cwe_ids"])

        for json_file in sorted(JSON_DIR.glob("*.json")):
            print(f"Reading {json_file.name}...")

            with open(json_file, "r", encoding="utf-8") as file:
                data = json.load(file)

            for item in data.get("vulnerabilities", []):
                cve = item.get("cve", {})

                cve_id = cve.get("id", "")
                description = get_description(cve)
                cwe_ids = get_unique_cwes(cve)

                if not cve_id or not description or not cwe_ids:
                    continue

                cve_year = int(cve_id.split("-")[1])

                if cve_year < START_YEAR:
                    continue

                if cve_id in seen_cve_ids:
                    continue

                seen_cve_ids.add(cve_id)

                writer.writerow(
                    [
                        cve_id,
                        description,
                        json.dumps(cwe_ids),
                    ]
                )

                saved_rows += 1

    print(f"CSV file created: {CSV_FILE}")
    print(f"Number of saved CVEs: {saved_rows}")


def main():
    download_json_files()
    unzip_json_files()
    create_csv()


if __name__ == "__main__":
    main()