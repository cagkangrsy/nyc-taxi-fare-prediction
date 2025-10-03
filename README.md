## NYC Taxi Fare Prediction
### Table of Contents
- [Key Features](#key-features)
- [Project Walkthrough](#project-walkthrough)
- [Project Structure](#project-structure)
- [Data Layout](#data-layout)
- [Scripts and CLI](#scripts-and-cli)
- [Serve the API](#serve-the-api-and-send-a-request)
- [Data Source](#data-source)

 Predict NYC yellow taxi trip fare from NYC TLC monthly data using a clean preprocessing pipeline and LightGBM optimized with Optuna. The repo also offers a small model zoo (Linear/Ridge/XGBoost/LightGBM/CatBoost) for quick baselines and comparisons.

### Key Features
- End-to-end pipeline: schema checks → column normalization → missing/outlier handling → feature engineering → dataset splits → preprocessing transformer → training/evaluation
- Strong feature set: time, holidays/long-weekend flags, cyclical encodings, geospatial distance via taxi zone centroids
- Simple CLI scripts for preprocessing and training

---
## Project Walkthrough

Read the end-to-end narrative with visuals and explanations. View on GitHub Pages (primary) or open the local HTML exports (fallback):

- EDA: [GitHub Pages: /eda.html](https://cagkangrsy.github.io/nyc-taxi-fare-prediction/eda.html) · [Local: docs/eda.html](docs/eda.html) · [Notebook: notebooks/eda.ipynb](notebooks/eda.ipynb)
- Preprocessing pipeline: [GitHub Pages: /preprocessing.html](https://cagkangrsy.github.io/nyc-taxi-fare-prediction/preprocessing.html) · [Local: docs/preprocessing.html](docs/preprocessing.html) · [Notebook: notebooks/preprocessing.ipynb](notebooks/preprocessing.ipynb)
- Model experiments and tuning: [GitHub Pages: /model_experiments.html](https://cagkangrsy.github.io/nyc-taxi-fare-prediction/model_experiments.html) · [Local: docs/model_experiments.html](docs/model_experiments.html) · [Notebook: notebooks/model_experiments.ipynb](notebooks/model_experiments.ipynb)

### Project Structure

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
├── docs
│   ├── index.html
│   ├── eda.html
│   ├── preprocessing.html
│   └── model_experiments.html
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
│   └── shape_data/
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

## Scripts and CLI

### 0) Download data

From the repo root:

```bash
python scripts/download_data.py  
```

This downloads all required months to `data/raw/`, fetches `taxi_zone_lookup.csv`, downloads `taxi_zones.zip`, unzips it into `data/raw/shape_data/`, and removes the zip.

### 1) Run preprocessing

From the repo root (explicit paths set):

```bash
python scripts/run_preprocess.py
```

This will create processed splits in `data/splits/` and save `artifacts/preprocess.joblib`.

Optional cleanup to remove intermediate stage folders after processing:

```bash
python scripts/run_preprocess.py --delete-intermediate
```

### 2) Train models

Evaluate baselines:

```bash
python scripts/run_training.py --mode evaluate --models ridge xgboost lightgbm catboost
```

Optimize LightGBM:

```bash
python scripts/run_training.py --mode optimize --n_trials 50 --early_stopping_rounds 200
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