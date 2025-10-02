from fastapi import FastAPI, HTTPException
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
import numpy as np
import pandas as pd
import joblib
from typing import Optional
import sys
import json

# Ensure project root is on sys.path so we can import `preprocess`
sys.path.append(str(Path(__file__).parent.parent))

from preprocess.feature_eng import build_lookup_maps, load_zone_centroids, fix_location_ids, add_zone_centroids, engineer_trip_features
from preprocess.features import prepare_features

class InputData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    VendorID: int
    tpep_pickup_datetime: datetime
    tpep_dropoff_datetime: datetime
    passenger_count: int
    trip_distance: float
    RatecodeID: float
    store_and_fwd_flag: str
    PULocationID: int
    DOLocationID: int
    payment_type: int


class PredictionResponse(BaseModel):
    total_amount: float
    currency: str
    model: str
    model_params: str


app = FastAPI(title="NYC Taxi Trip Fare Prediction")

# Globals loaded at startup
ARTIFACT_DIR = Path("../artifacts")
DATA_DIR = Path("../data")
preprocess = None
model = None
borough_map = None
service_zone_map = None
zones_keep = None
model_name = "Unknown"
model_params = "Unknown"


def load_artifacts():
    global preprocess, model, borough_map, service_zone_map, zones_keep, model_name, model_params

    pre_path = ARTIFACT_DIR / "preprocess.joblib"
    model_path = ARTIFACT_DIR / "model_experiments" / "final_eval" / "model.joblib"

    if not pre_path.exists():
        raise FileNotFoundError(f"Missing preprocess artifact: {pre_path}")
    preprocess = joblib.load(pre_path)

    if model_path is None:
        raise FileNotFoundError("No model artifact found under artifacts/")
    model = joblib.load(model_path)

    # Try to infer model info
    sel_path = ARTIFACT_DIR / "model_experiments" / "optuna_lgbm" / "best_params.json"
    if sel_path.exists():
        try:
            with open(sel_path, "r", encoding="utf-8") as f:
                best_params = json.load(f)
            model_name = "LightGBM"
            # Store params as a compact JSON string for visibility in responses/logs
            model_params = json.dumps(best_params, separators=(",", ":"))
        except Exception:
            # Fall back gracefully if params cannot be parsed
            model_name = "LightGBM"
            model_params = "unknown"
    else:
        model_name = "LightGBM"
        model_params = "unknown"


    # Build lookup maps for borough/service zones
    lookup_csv = DATA_DIR / "raw" / "taxi_zone_lookup.csv"
    if not lookup_csv.exists():
        raise FileNotFoundError(f"Missing lookup csv: {lookup_csv}")
    borough_map, service_zone_map = build_lookup_maps(lookup_csv)

    # Load zone centroids for distance features
    shp_path = DATA_DIR / "raw" / "shape_data" / "taxi_zones.shp"
    if shp_path.exists():
        zones_keep = load_zone_centroids(shp_path)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.on_event("startup")
def _startup_event():
    load_artifacts()


@app.post("/predict", response_model=PredictionResponse)
def predict(item: InputData) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if preprocess is None:
        raise HTTPException(status_code=503, detail="Artifacts not loaded")

    # Build a single-row DataFrame from input
    row = {
        "VendorID": item.VendorID,
        "tpep_pickup_datetime": pd.to_datetime(item.tpep_pickup_datetime),
        "tpep_dropoff_datetime": pd.to_datetime(item.tpep_dropoff_datetime),
        "passenger_count": item.passenger_count,
        "trip_distance": float(item.trip_distance),
        "RatecodeID": float(item.RatecodeID),
        "store_and_fwd_flag": item.store_and_fwd_flag,
        "PULocationID": int(item.PULocationID),
        "DOLocationID": int(item.DOLocationID),
        "payment_type": int(item.payment_type),
    }
    df = pd.DataFrame([row])

    # Apply the same feature engineering steps as training
    try:
        df = fix_location_ids(df)
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="Input filtered by location ID rules")

        if zones_keep is not None:
            df = add_zone_centroids(df, zones_keep)
        else:
            raise HTTPException(
                status_code=400,
                detail="Zone centroids unavailable; cannot compute haversine distance."
            )

        df_feat = engineer_trip_features(df, borough_map, service_zone_map)
        if len(df_feat) == 0:
            raise HTTPException(status_code=400, detail="Input filtered by trip duration rules")

        # Require realistic haversine distance
        if "haversine_distance" not in df_feat.columns or df_feat["haversine_distance"].isna().any():
            raise HTTPException(
                status_code=400,
                detail="Cannot compute haversine distance for provided LocationIDs; verify PULocationID/DOLocationID are valid."
            )

        X = prepare_features(df_feat)

        X_t = preprocess.transform(X)
        y_pred = model.predict(X_t)
        pred = round(float(np.asarray(y_pred).ravel()[0]),2)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    return PredictionResponse(
        total_amount=pred,
        currency="USD",
        model=model_name,
        model_params=model_params,
    )