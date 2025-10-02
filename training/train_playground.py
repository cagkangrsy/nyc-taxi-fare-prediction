import json
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)


def fit_time_predict(name, est, Xtr, ytr, Xva, yva):
    if name == "MeanBaseline":
        t0 = time.time()
        yhat = np.full_like(yva, ytr.mean(), dtype=float)
        return dict(
            Model=name,
            MAE=mean_absolute_error(yva, yhat),
            RMSE=root_mean_squared_error(yva, yhat),
            R2=r2_score(yva, yhat),
            Fit_s=0.0,
            Pred_s=time.time() - t0,
        )

    t0 = time.time()
    est.fit(Xtr, ytr)
    fit_s = time.time() - t0

    t1 = time.time()
    yhat = est.predict(Xva)
    pred_s = time.time() - t1

    return dict(
        Model=name,
        MAE=mean_absolute_error(yva, yhat),
        RMSE=root_mean_squared_error(yva, yhat),
        R2=r2_score(yva, yhat),
        Fit_s=fit_s,
        Pred_s=pred_s,
    )

def evaluate_model_zoo(models, X_train, y_train, X_val, y_val, exp_dir: Path):
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Run evaluations; accept entries as (name, estimator) or (name, estimator, is_baseline)
    normalized = []
    for entry in models:
        if len(entry) == 3:
            model_name, estimator, _ = entry
        elif len(entry) == 2:
            model_name, estimator = entry
        else:
            raise ValueError("Each model must be a 2- or 3-tuple: (name, estimator[, flag])")
        normalized.append((model_name, estimator))

    # Ensure MeanBaseline runs first once
    seen = set(name for name, _ in normalized)
    ordered = [("MeanBaseline", None)] + [pair for pair in normalized if pair[0] != "MeanBaseline"]

    results = []
    for (model_name, estimator) in ordered:
        print(f"\n[Model] {model_name} — starting...", flush=True)
        res = fit_time_predict(model_name, estimator, X_train, y_train, X_val, y_val)
        results.append(res)
        print(
            (
                f"[Done] {model_name}: "
                f"MAE={res['MAE']:.3f}, RMSE={res['RMSE']:.3f}, R2={res['R2']:.4f}, "
                f"Fit_s={res['Fit_s']:.2f}, Pred_s={res['Pred_s']:.2f}"
            ),
            flush=True,
        )

    df_val = (
        pd.DataFrame(results).sort_values("RMSE", ascending=True).reset_index(drop=True)
    )

    # Console output
    print("\nValidation Performance:")
    print(df_val.to_string(index=False, formatters={
        "MAE": lambda v: f"{v:.3f}",
        "RMSE": lambda v: f"{v:.3f}",
        "R2": lambda v: f"{v:.4f}",
        "Fit_s": lambda v: f"{v:.2f}",
        "Pred_s": lambda v: f"{v:.2f}",
    }))

    # Charts
    for metric, title, fname in [
        ("RMSE", "Validation RMSE", "val_rmse_bar.png"),
        ("MAE", "Validation MAE", "val_mae_bar.png"),
    ]:
        plt.figure(figsize=(7, 4))
        plt.barh(df_val["Model"], df_val[metric])
        plt.gca().invert_yaxis()
        for i, v in enumerate(df_val[metric]):
            plt.text(v, i, f"{v:.3f}", va="center")
        plt.title(title)
        plt.xlabel(metric)
        plt.ylabel("Model")
        plt.tight_layout()
        plt.savefig(exp_dir / fname, dpi=150)
        plt.close()

    df_val.to_csv(exp_dir / "val_results.csv", index=False)

    with open(exp_dir / "val_results.json", "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in row.items()
                }
                for row in results
            ],
            f,
            indent=2,
        )

    best_row = df_val.iloc[0]
    best_name = best_row["Model"]
    if best_name == "MeanBaseline" and len(df_val) > 1:
        best_row = df_val.iloc[1]
        best_name = best_row["Model"]

    print("\nBest model (by RMSE):", best_name)
    print("Artifacts saved in:", exp_dir)
    print("Files: val_results.csv, val_results.json, val_rmse_bar.png, val_mae_bar.png")

    return df_val