from pathlib import Path
import joblib
import pandas as pd


def build_transform_splits(data_dir, artifact_dir):
    DATA_SPLIT_DIR = Path(data_dir)
    BASE_ARTIFACTS_DIR = Path(artifact_dir)

    TARGET = "total_amount"

    FP_TRAIN = DATA_SPLIT_DIR / "train.parquet"
    FP_VAL = DATA_SPLIT_DIR / "val.parquet"
    FP_TEST = DATA_SPLIT_DIR / "test.parquet"

    # Load fitted preprocessing
    preprocess = joblib.load(BASE_ARTIFACTS_DIR / "preprocess.joblib")

    # Columns the transformer was fitted on
    INPUT_COLS = []
    for name, trans, cols in preprocess.transformers_:
        if name == "remainder":
            continue
        INPUT_COLS.extend(list(cols))

    use_cols = sorted(set(INPUT_COLS + [TARGET]))
    df_train = pd.read_parquet(FP_TRAIN, columns=use_cols)
    df_val = pd.read_parquet(FP_VAL, columns=use_cols)
    df_test = pd.read_parquet(FP_TEST, columns=use_cols)

    X_train = preprocess.transform(df_train[INPUT_COLS])
    X_val = preprocess.transform(df_val[INPUT_COLS])
    X_test = preprocess.transform(df_test[INPUT_COLS])

    y_train = df_train[TARGET].to_numpy()
    y_val = df_val[TARGET].to_numpy()
    y_test = df_test[TARGET].to_numpy()

    print("  X_train:", X_train.shape, " y_train:", y_train.shape)
    print("  X_val:  ", X_val.shape, " y_val:  ", y_val.shape)
    print("  X_test: ", X_test.shape, " y_test: ", y_test.shape)

    return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]