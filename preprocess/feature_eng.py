from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import holidays
import glob
from typing import Optional


def load_zone_centroids(zones_shapefile: Path | str) -> pd.DataFrame:
    """
    Load Taxi Zones shapefile and compute WGS84 centroids per `LocationID`.

    Parameters
    ----------
    zones_shapefile : Path | str
        Path to the Taxi Zones shapefile.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns `LocationID`, `lon`, `lat`.
    """
    zones = gpd.read_file(zones_shapefile).to_crs(2263)
    cent_wgs = zones.geometry.centroid.to_crs(4326)
    zones["lon"], zones["lat"] = cent_wgs.x, cent_wgs.y
    return zones[["LocationID", "lon", "lat"]].copy()


def fix_location_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize and filter invalid TLC location IDs.

    - Replace `LocationID` 57 with 56 in PU and DO columns.
    - Drop rows where PU or DO is in the known-bad set {264, 265}.
    """
    df = df.copy()
    df["PULocationID"] = df["PULocationID"].replace(57, 56)
    df["DOLocationID"] = df["DOLocationID"].replace(57, 56)
    bad_ids = [264, 265]
    mask_bad = df["PULocationID"].isin(bad_ids) | df["DOLocationID"].isin(bad_ids)
    return df.loc[~mask_bad].copy()


def add_zone_centroids(df: pd.DataFrame, zones_keep: pd.DataFrame) -> pd.DataFrame:
    """
    Join centroid coordinates for PU/DO `LocationID` onto the dataframe.

    Adds columns: `pickup_lon`, `pickup_lat`, `dropoff_lon`, `dropoff_lat`.
    """
    df = df.merge(
        zones_keep.rename(columns={"lon": "pickup_lon", "lat": "pickup_lat"}),
        left_on="PULocationID",
        right_on="LocationID",
        how="left",
    ).drop(columns="LocationID")
    df = df.merge(
        zones_keep.rename(columns={"lon": "dropoff_lon", "lat": "dropoff_lat"}),
        left_on="DOLocationID",
        right_on="LocationID",
        how="left",
    ).drop(columns="LocationID")
    return df


def build_lookup_maps(lookup_csv: Path | str) -> tuple[dict[int, str], dict[int, str]]:
    """
    Build `LocationID` -> borough and service_zone mapping dictionaries.
    """
    csv_path = Path(lookup_csv)
    tz = pd.read_csv(csv_path)
    tz = tz.rename(columns={c: c.strip() for c in tz.columns})
    tz["LocationID"] = tz["LocationID"].astype(int)
    borough_map = dict(zip(tz["LocationID"], tz["Borough"]))
    service_zone_map = dict(zip(tz["LocationID"], tz["service_zone"]))
    return borough_map, service_zone_map


def haversine_distance(lon1, lat1, lon2, lat2) -> np.ndarray:
    """
    Compute great-circle distance (km) between two lon/lat points.
    """
    R = 6371.0
    lon1, lat1, lon2, lat2 = np.radians([lon1, lat1, lon2, lat2])
    dlon, dlat = lon2 - lon1, lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * R * np.arcsin(np.sqrt(a))


def engineer_trip_features(
    df: pd.DataFrame,
    borough_map: dict[int, str],
    service_zone_map: dict[int, str],
) -> pd.DataFrame:
    """
    Engineer time, holiday, geographic, and categorical context features.

    Input must contain pickup/dropoff timestamps; optional PU/DO LocationIDs and
    centroid coordinates enable additional features.

    Returns a copy with engineered features and drops raw timestamp/location ID columns.
    """
    df = df.copy()

    df["trip_duration"] = (
        pd.to_datetime(df["tpep_dropoff_datetime"], errors="coerce")
        - pd.to_datetime(df["tpep_pickup_datetime"], errors="coerce")
    ).dt.total_seconds() / 60

    lower_thresh, upper_thresh = 1, 180
    mask_outliers = (df["trip_duration"] < lower_thresh) | (
        df["trip_duration"] > upper_thresh
    )
    df = df.loc[~mask_outliers].copy()

    df["pickup_hour"] = df["tpep_pickup_datetime"].dt.hour
    df["dropoff_hour"] = df["tpep_dropoff_datetime"].dt.hour
    df["pickup_dayofweek"] = df["tpep_pickup_datetime"].dt.dayofweek
    df["pickup_month"] = df["tpep_pickup_datetime"].dt.month
    df["pickup_minute"] = df["tpep_pickup_datetime"].dt.minute

    df["pickup_hour_sin"] = np.sin(2 * np.pi * df["pickup_hour"] / 24)
    df["pickup_hour_cos"] = np.cos(2 * np.pi * df["pickup_hour"] / 24)
    df["dropoff_hour_sin"] = np.sin(2 * np.pi * df["dropoff_hour"] / 24)
    df["dropoff_hour_cos"] = np.cos(2 * np.pi * df["dropoff_hour"] / 24)

    df["dow_sin"] = np.sin(2 * np.pi * df["pickup_dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["pickup_dayofweek"] / 7)

    df["month_sin"] = np.sin(2 * np.pi * (df["pickup_month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["pickup_month"] - 1) / 12)

    years_needed = df["tpep_pickup_datetime"].dt.year.dropna().unique().astype(int)
    us_holidays = holidays.US(years=years_needed.tolist(), state="NY")

    df["pickup_date"] = df["tpep_pickup_datetime"].dt.date
    df["is_holiday"] = df["pickup_date"].isin(us_holidays).astype(int)

    d = df["pickup_dayofweek"]
    dates_ts = pd.to_datetime(df["pickup_date"])
    fri_before_mon = (d == 4) & (
        (dates_ts + pd.Timedelta(days=3)).dt.date.isin(us_holidays)
    )
    mon_after_fri = (d == 0) & (
        (dates_ts - pd.Timedelta(days=3)).dt.date.isin(us_holidays)
    )
    df["is_long_weekend"] = (fri_before_mon | mon_after_fri).astype(int)

    h, m = df["pickup_hour"], df["pickup_minute"]
    weekday_morning = (d < 5) & (
        ((h == 6) & (m >= 30)) | ((h >= 7) & (h < 10))
    )
    weekday_evening = (d < 5) & (h >= 16) & (h < 20)
    late_night = (d.isin([3, 4, 5])) & ((h >= 22) | (h < 2))
    friday_getaway = (
        (d == 4)
        & (h >= 12)
        & (h < 20)
        & ((df["is_holiday"] == 1) | (df["is_long_weekend"] == 1))
    )

    df["pickup_period"] = np.select(
        [friday_getaway, weekday_morning, weekday_evening, late_night],
        ["friday_getaway", "weekday_morning", "weekday_evening", "late_night"],
        default="other",
    )

    df["is_weekend"] = df["pickup_dayofweek"].isin([5, 6]).astype(int)
    df["is_rush_hour"] = (weekday_morning | weekday_evening).astype(int)
    df["is_nightlife"] = late_night.astype(int)
    df["is_friday_getaway"] = friday_getaway.astype(int)

    if {"pickup_lon", "pickup_lat", "dropoff_lon", "dropoff_lat"}.issubset(df.columns):
        df["haversine_distance"] = haversine_distance(
            df["pickup_lon"], df["pickup_lat"], df["dropoff_lon"], df["dropoff_lat"]
        )

    if "PULocationID" in df.columns:
        df["pickup_borough"] = df["PULocationID"].map(borough_map).fillna("Unknown")
        df["pickup_service_zone"] = (
            df["PULocationID"].map(service_zone_map).fillna("Unknown")
        )
    else:
        df["pickup_borough"] = "Unknown"
        df["pickup_service_zone"] = "Unknown"

    if "DOLocationID" in df.columns:
        df["dropoff_borough"] = df["DOLocationID"].map(borough_map).fillna("Unknown")
        df["dropoff_service_zone"] = (
            df["DOLocationID"].map(service_zone_map).fillna("Unknown")
        )
    else:
        df["dropoff_borough"] = "Unknown"
        df["dropoff_service_zone"] = "Unknown"

    df = df.drop(
        columns=[
            "pickup_hour",
            "dropoff_hour",
            "pickup_dayofweek",
            "pickup_month",
            "pickup_minute",
            "tpep_pickup_datetime",
            "tpep_dropoff_datetime",
            "PULocationID",
            "DOLocationID",
        ],
        errors="ignore",
    )

    return df


def build_and_save_features(
    src: Path | str = Path("../data/outliers_removed"),
    dst: Path | str = Path("../data/features_built"),
    lookup_csv: Path | str = Path("../data/raw/taxi_zone_lookup.csv"),
    zones_shapefile: Optional[Path | str] = Path("../data/raw/shape_data/taxi_zones.shp"),
) -> Path:
    """
    Build feature-engineered parquet files from cleaned monthly inputs and save them.

    Parameters
    ----------
    src : Path | str, default ../data/outliers_removed
        Source directory containing monthly parquet files after outlier removal.
    dst : Path | str, default ../data/features_built
        Destination directory to write feature-engineered parquet files.
    lookup_csv : Path | str, default ../data/raw/taxi_zone_lookup.csv
        CSV file for mapping `LocationID` to borough and service zone.
    zones_shapefile : Optional[Path | str], default ../data/raw/shape_data/taxi_zones.shp
        Optional path to a Taxi Zones shapefile. If provided, centroid coordinates
        will be merged to enable distance features; otherwise, geographic features
        that rely on coordinates are skipped if coordinates are absent.

    Notes
    -----
    - This function applies `fix_location_ids`, optional centroid enrichment via
      `add_zone_centroids`, and then computes time- and holiday-based features via
      `engineer_trip_features`.
    - Files are processed per-month and saved with the same filenames into `dst`.
    """
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(src / "yellow_tripdata_20*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {src}")

    borough_map, service_zone_map = build_lookup_maps(lookup_csv)

    zones_keep = None
    if zones_shapefile is not None and Path(zones_shapefile).exists():
        try:
            zones_keep = load_zone_centroids(zones_shapefile)
        except Exception as e:
            print(f"Warning: failed to load zones shapefile {zones_shapefile}: {e}")
            zones_keep = None

    for fp in files:
        df = pd.read_parquet(fp)
        df = fix_location_ids(df)
        if zones_keep is not None:
            df = add_zone_centroids(df, zones_keep)
        df_feat = engineer_trip_features(df, borough_map, service_zone_map)
        if "haversine_distance" not in df_feat.columns:
            # Ensure column exists to satisfy downstream feature expectations
            df_feat["haversine_distance"] = 0.0

        out_fp = dst / Path(fp).name
        df_feat.to_parquet(out_fp, index=False, compression="zstd")
        # Keep quiet by default; logging handled at script level

    return dst