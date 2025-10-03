## NYC Taxi Fare Prediction

 Predict NYC yellow taxi trip fare from NYC TLC monthly data using a clean preprocessing pipeline and LightGBM optimized with Optuna. The repo also offers a small model zoo (Linear/Ridge/XGBoost/LightGBM/CatBoost) for quick baselines and comparisons.

### Key Features
- End-to-end pipeline: schema checks → column normalization → missing/outlier handling → feature engineering → dataset splits → preprocessing transformer → training/evaluation
- Strong feature set: time, holidays/long-weekend flags, cyclical encodings, geospatial distance via taxi zone centroids
- Simple CLI scripts for preprocessing and training

---

## Project Structure

```
.
├── preprocess
│   ├── feature_eng.py
│   ├── features.py
│   ├── missing_val.py
│   ├── normalize.py
│   ├── outlier.py
│   └── __init__.py
├── training
│   ├── build_transform.py
│   ├── optimize.py
│   └── train_playground.py
├── scripts
│   ├── run_preprocess.py
│   ├── run_training.py
│   ├── download_data.py
│   └── app.py
├── data
│   ├── raw/
│   ├── columns_normalized/
│   ├── month_filtered/
│   ├── missing_handled/
│   ├── outliers_removed/
│   ├── features_built/
│   └── splits/
├── artifacts
│   ├── final/
│   ├── model_experiments/
│   └── optimize/
├── tests
│   ├── test_app.py
│   ├── test_preprocess.py
│   └── test_training.py
├── notebooks
│   ├── eda.ipynb
│   ├── preprocessing.ipynb
│   └── model_experiments.ipynb
├── LICENSE
└── README.md
```

## Data Layout

Place raw and processed files under:

```
data/
├── raw
│   ├── yellow_tripdata_YYYY-MM.parquet
│   ├── taxi_zone_lookup.csv
│   └── shape/
│       └── taxi_zones.shp
├── columns_normalized/
├── month_filtered/
├── missing_handled/
├── outliers_removed/
├── features_built/
└── splits/
    ├── train.parquet
    ├── val.parquet
    └── test.parquet
```

## Usage

### 0) Download data

From the repo root:

```bash
python scripts/download_data.py  
```

This downloads all required months to `data/raw/`, fetches `taxi_zone_lookup.csv`, downloads `taxi_zones.zip`, unzips it into `data/raw/shape/`, and removes the zip.

### 1) Run preprocessing

From the repo root (explicit paths set):

```bash
python scripts/run_preprocess.py
```

This will create processed splits in `data/splits/` and save `artifacts/preprocess.joblib`.

### 2) Train models

Evaluate baselines:

```bash
python scripts/run_training.py --mode evaluate --models ridge xgboost lightgbm catboost
```

Optimize LightGBM:

```bash
python scripts/run_training.py --mode optimize --n_trials 50 --early_stopping_rounds 200 \
```

Export final model and test metrics (requires `artifacts/optimize/best_params.json`):

```bash
python scripts/run_training.py --mode test
```

Artifacts will be under `artifacts/optimize/` and `artifacts/final/`.

### 3) Serve the API and send a request

The FastAPI app expects `artifacts/` and `data/` one directory above the server’s working directory. Run it from `scripts/` so the defaults resolve to `../artifacts` and `../data`:

```bash
cd scripts
uvicorn app:app --reload --port 8000
```

Health check:

```bash
curl http://localhost:8000/healthz
```

Example prediction request (curl):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "VendorID": 2,
    "tpep_pickup_datetime": "2025-01-15T08:15:00",
    "tpep_dropoff_datetime": "2025-01-15T08:35:00",
    "passenger_count": 1,
    "trip_distance": 4.2,
    "RatecodeID": 1,
    "store_and_fwd_flag": "N",
    "PULocationID": 142,
    "DOLocationID": 238,
    "payment_type": 1
  }'
```

## Data Source
Data: [NYC TLC Trip Record Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)