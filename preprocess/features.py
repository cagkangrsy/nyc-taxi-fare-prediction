from pathlib import Path
from typing import Sequence, Tuple, Optional

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


# Config
TARGET = "total_amount"

# Feature groups
NUMERIC: Sequence[str] = [
    "trip_distance",
    "trip_duration",
    "haversine_distance",
    "pickup_lon",
    "pickup_lat",
    "dropoff_lon",
    "dropoff_lat",
    "pickup_hour_sin",
    "pickup_hour_cos",
    "dropoff_hour_sin",
    "dropoff_hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "passenger_count",
]

LOW_CARD_CAT: Sequence[str] = [
    "VendorID",
    "RatecodeID",
    "store_and_fwd_flag",
    "payment_type",
    "pickup_borough",
    "dropoff_borough",
    "pickup_service_zone",
    "dropoff_service_zone",
    "pickup_period",
    "is_holiday",
    "is_long_weekend",
    "is_weekend",
    "is_rush_hour",
    "is_nightlife",
    "is_friday_getaway",
]

# Columns to drop before feeding the transformer
DROP_COLS: Sequence[str] = ["pickup_date"]


def build_preprocess(
    numeric_features: Optional[Sequence[str]] = None,
    categorical_features: Optional[Sequence[str]] = None,
) -> ColumnTransformer:
    """
    Create a preprocessing ColumnTransformer for numeric scaling and categorical OHE.

    Parameters
    ----------
    numeric_features : Optional[Sequence[str]]
        Numeric columns to scale. Defaults to built-in NUMERIC list if None.
    categorical_features : Optional[Sequence[str]]
        Categorical columns to one-hot encode. Defaults to LOW_CARD_CAT if None.
    """
    numeric = list(numeric_features) if numeric_features is not None else list(NUMERIC)
    cats = list(categorical_features) if categorical_features is not None else list(LOW_CARD_CAT)

    numeric_tf = Pipeline([("scale", RobustScaler())])
    cat_tf = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    return ColumnTransformer(
        transformers=[
            ("num", numeric_tf, numeric),
            ("cat", cat_tf, cats),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Select and return model feature columns from an engineered dataframe.
    """
    use_cols = [c for c in (list(NUMERIC) + list(LOW_CARD_CAT)) if c in df.columns]
    out = df[use_cols].copy()
    for c in DROP_COLS:
        if c in out.columns:
            out = out.drop(columns=c)
    return out


def split_xy(df: pd.DataFrame, target: str = TARGET) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into X features and y target.
    """
    X = prepare_features(df)
    y = df[target]
    return X, y


def fit_preprocess(preprocess: ColumnTransformer, X_train: pd.DataFrame) -> ColumnTransformer:
    """
    Fit the preprocessing pipeline on training features.
    """
    preprocess.fit(X_train)
    return preprocess


def transform(preprocess: ColumnTransformer, X: pd.DataFrame):
    """
    Transform features using a fitted preprocessing pipeline.
    """
    return preprocess.transform(X)


def get_feature_names(preprocess: ColumnTransformer):
    """
    Get output feature names after preprocessing.
    """
    return preprocess.get_feature_names_out()


def save_preprocess(preprocess: ColumnTransformer, artifacts_dir: Path | str = Path("../artifacts")) -> Path:
    """
    Persist the fitted preprocessing pipeline to `artifacts_dir/preprocess.joblib`.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "preprocess.joblib"
    joblib.dump(preprocess, out)
    return out


def save_feature_names(feature_names, artifacts_dir: Path | str = Path("../artifacts")) -> Path:
    """
    Persist feature names to `artifacts_dir/feature_names.joblib`.
    """
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out = artifacts_dir / "feature_names.joblib"
    joblib.dump(feature_names, out)
    return out


def build_and_save_preprocess(
    train_path: Path | str = Path("../data/splits/train.parquet"),
    artifacts_dir: Path | str = Path("../artifacts"),
) -> Tuple[Path, Path]:
    """
    Fit preprocessing on the training split and save both the pipeline and names.

    Parameters
    ----------
    train_path : Path | str, default ../data/splits/train.parquet
        Path to the training dataframe parquet with engineered features and target.
    artifacts_dir : Path | str, default ../artifacts
        Directory where artifacts will be written.

    Returns
    -------
    Tuple[Path, Path]
        Paths to saved preprocess pipeline and feature names joblib files.
    """
    train_path = Path(train_path)
    df_train = pd.read_parquet(train_path)
    X_train, _ = split_xy(df_train)
    pre = build_preprocess()
    pre = fit_preprocess(pre, X_train)
    feature_names = get_feature_names(pre)
    p_path = save_preprocess(pre, artifacts_dir)
    f_path = save_feature_names(feature_names, artifacts_dir)
    return p_path, f_path