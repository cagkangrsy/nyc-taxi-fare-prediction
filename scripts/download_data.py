from __future__ import annotations
import os
from urllib.request import urlopen
import zipfile
from os.path import abspath, dirname, join

# Fixed months (2024-01 to 2025-07)
MONTHS = [
    "2024-01","2024-02","2024-03","2024-04","2024-05","2024-06","2024-07",
    "2024-08","2024-09","2024-10","2024-11","2024-12",
    "2025-01","2025-02","2025-03","2025-04","2025-05","2025-06","2025-07",
]

BASE_TRIP = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{}.parquet"
CSV_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
ZIP_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"

# Resolve repo root so outputs always go to <repo>/data/raw even if run from scripts/
SCRIPT_DIR = dirname(abspath(__file__))
REPO_ROOT = dirname(SCRIPT_DIR)
RAW_DIR = join(REPO_ROOT, "data", "raw")


def fetch(url: str, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with urlopen(url) as r, open(out_path, "wb") as f:
        f.write(r.read())


def main() -> None:
    # Trips
    print(f"Downloading {len(MONTHS)} monthly parquet files...")
    for m in MONTHS:
        out = join(RAW_DIR, f"yellow_tripdata_{m}.parquet")
        print(f"[DONE] {m}")
        fetch(BASE_TRIP.format(m), out)

    # Taxi zone CSV
    print("Downloading taxi_zone_lookup.csv ...")
    fetch(CSV_URL, join(RAW_DIR, "taxi_zone_lookup.csv"))

    # Taxi zones ZIP + unzip + cleanup
    zip_out = join(RAW_DIR, "taxi_zones.zip")
    unzip_dir = join(RAW_DIR, "shape")
    print("Downloading taxi_zones.zip ...")
    fetch(ZIP_URL, zip_out)
    os.makedirs(unzip_dir, exist_ok=True)
    print("Unzipping ...")
    with zipfile.ZipFile(zip_out, "r") as zf:
        zf.extractall(unzip_dir)
    os.remove(zip_out)
    print("Done.")


if __name__ == "__main__":
    main()