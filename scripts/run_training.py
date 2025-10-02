import argparse
import json
import random
import sys
import warnings
from pathlib import Path
import catboost as cat
import joblib
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from training import build_transform_splits, optimize_lgbm
from training.train_playground import evaluate_model_zoo
warnings.filterwarnings("ignore")

# Add parent directory to path so we can import preprocess module
sys.path.append(str(Path(__file__).parent.parent))


MODEL_CONFIG_DICT = {
"ridge" : dict(alpha=1.0,
               random_state=None),
"xgboost" : dict(objective="reg:squarederror",
             n_estimators=700,
             max_depth=8,
             learning_rate=0.08,
             subsample=0.9,
             colsample_bytree=0.9,
             tree_method="hist",
             random_state=None,
             n_jobs=-1,
             verbosity=0),
"lightgbm" : dict(n_estimators=1200,
             learning_rate=0.05,
             random_state=None,
             n_jobs=-1,
             verbosity=-1),
"catboost" : dict(iterations=1200,
             learning_rate=0.05,
             depth=8,
             random_seed=None,
             verbose=False)
}


warnings.filterwarnings("ignore")

def main(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)

    MODEL_CONFIG_DICT["xgboost"]["random_state"] = seed
    MODEL_CONFIG_DICT["lightgbm"]["random_state"] = seed
    MODEL_CONFIG_DICT["catboost"]["random_seed"] = seed
    splits = build_transform_splits(args.data_root, args.artifact_root)

    X_train, y_train = splits[0]
    X_val, y_val = splits[1]
    X_test, y_test = splits[2]

    if args.mode == "evaluate":
        models = [("MeanBaseline", None, True)]
        for model in args.models:
            if model == 'linear_reg':
                models.append(("LinearRegression", LinearRegression()))
            if model == 'ridge':
                models.append(("Ridge", Ridge(alpha=1.0, random_state=42)))
            if model == 'xgboost':
                models.append(("XGBoost", xgb.XGBRegressor(**MODEL_CONFIG_DICT["xgboost"])) )
            if model == 'lightgbm':
                models.append(("LightGBM", lgb.LGBMRegressor(**MODEL_CONFIG_DICT["lightgbm"])) )
            if model == 'catboost':
                models.append(("CatBoost", cat.CatBoostRegressor(**MODEL_CONFIG_DICT["catboost"])) )

        exp_dir = Path(args.artifact_root) / "model_experiments"
        evaluate_model_zoo(models, X_train, y_train, X_val, y_val, exp_dir)
    elif args.mode == "optimize":
        if args.opt_model.lower() != "lightgbm":
            raise ValueError("Only LightGBM optimization is implemented. Use --opt_model lightgbm")
        best_value, best_params, exp_dir = optimize_lgbm(
            X_train,
            y_train,
            X_val,
            y_val,
            artifact_root=args.artifact_root,
            n_trials=args.n_trials,
            timeout=args.timeout,
            seed=args.seed,
            early_stopping_rounds=args.early_stopping_rounds,
        )
        print("\nOptimization finished.")
        print("Best RMSE:", best_value)
        print("Best params:", json.dumps(best_params, indent=2))
        print("Artifacts saved to:", exp_dir)
        
    elif args.mode == "test":
        # Compare training on TRAIN vs TRAIN+VAL using saved best LightGBM params; evaluate both on TEST

        exp_dir = Path(args.artifact_root) / "model_experiments" / "final_eval"
        exp_dir.mkdir(parents=True, exist_ok=True)

        params_path = (
            Path(args.artifact_root) / "model_experiments" / "optuna_lgbm" / "best_params.json"
        )
        if not params_path.exists():
            raise FileNotFoundError(
                f"Could not find best_params.json at {params_path}. Run --mode optimize first."
            )

        print("[Test] Using params from:", params_path)
        with open(params_path, "r", encoding="utf-8") as f:
            best_params = json.load(f)

        best_params.update({"random_state": args.seed, "n_jobs": -1, "verbosity": -1})

        # 1) Train on TRAIN only
        print("\n[Test] Training on TRAIN only...")
        model_train_only = lgb.LGBMRegressor(**best_params)
        model_train_only.fit(X_train, y_train)
        y_pred_train_only = model_train_only.predict(X_test)
        rmse_train_only = root_mean_squared_error(y_test, y_pred_train_only)
        mae_train_only = mean_absolute_error(y_test, y_pred_train_only)
        r2_train_only = r2_score(y_test, y_pred_train_only)
        print("[Test] Results (TRAIN only): RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(
            rmse_train_only, mae_train_only, r2_train_only
        ))

        # Save predictions and metrics
        np.save(exp_dir / "y_test_pred_train.npy", y_pred_train_only)
        with open(exp_dir / "metrics_test_train.json", "w", encoding="utf-8") as f:
            json.dump({
                "mode": "train_only",
                "split": "TEST",
                "rmse": float(rmse_train_only),
                "mae": float(mae_train_only),
                "r2": float(r2_train_only),
                "params_path": str(params_path),
            }, f, indent=2)

        # 2) Train on TRAIN+VAL
        print("\n[Test] Training on TRAIN+VAL...")
        X_trval = np.concatenate([X_train, X_val], axis=0)
        y_trval = np.concatenate([y_train, y_val], axis=0)
        model_trainval = lgb.LGBMRegressor(**best_params)
        model_trainval.fit(X_trval, y_trval)
        y_pred_trainval = model_trainval.predict(X_test)
        rmse_trainval = root_mean_squared_error(y_test, y_pred_trainval)
        mae_trainval = mean_absolute_error(y_test, y_pred_trainval)
        r2_trainval = r2_score(y_test, y_pred_trainval)
        print("[Test] Results (TRAIN+VAL): RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(
            rmse_trainval, mae_trainval, r2_trainval
        ))

        # Save model, predictions and metrics for TRAIN+VAL
        model_path = exp_dir / "model.joblib"
        joblib.dump(model_trainval, model_path)
        np.save(exp_dir / "y_test_pred_trainval.npy", y_pred_trainval)
        with open(exp_dir / "metrics_test_trainval.json", "w", encoding="utf-8") as f:
            json.dump({
                "mode": "train_val",
                "split": "TEST",
                "rmse": float(rmse_trainval),
                "mae": float(mae_trainval),
                "r2": float(r2_trainval),
                "model_path": str(model_path),
                "params_path": str(params_path),
            }, f, indent=2)

        # Comparison summary
        print("\n[Test] Comparison (lower is better for RMSE/MAE):")
        print("   TRAIN only  : RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(
            rmse_train_only, mae_train_only, r2_train_only
        ))
        print("   TRAIN+VAL   : RMSE={:.4f} MAE={:.4f} R2={:.4f}".format(
            rmse_trainval, mae_trainval, r2_trainval
        ))
        better = "TRAIN+VAL" if rmse_trainval <= rmse_train_only else "TRAIN only"
        print(f"   => Better by RMSE: {better}")

        print("\nArtifacts saved in:", exp_dir)
        print("Files: model.joblib, metrics_test_train.json, metrics_test_trainval.json,"
              " y_test_pred_train.npy, y_test_pred_trainval.npy")
    else:
        raise ValueError(f"Unknown mode: {args.mode}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NYC taxi training pipeline")
    parser.add_argument("--artifact_root", type=str, default="../artifacts",
                        help="Path to artifacts directory")
    parser.add_argument("--data_root", type=str, default="../data/splits",
                        help="Path to the processed data splits directory")
    parser.add_argument("--mode", type=str, choices=["evaluate", "optimize", "test"], default="test",
                        help="Run evaluation of models or hyperparameter optimization")
    parser.add_argument("--models", nargs="+", default=["ridge", "xgboost", "lightgbm", "catboost"],
                        help="Models to evaluate (linear_reg ridge xgboost lightgbm catboost). Used in evaluate mode.")
    parser.add_argument("--opt_model", type=str, default="lightgbm",
                        help="Which model to optimize (currently only 'lightgbm')")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of Optuna trials for optimization mode")
    parser.add_argument("--timeout", type=int, default=None, help="Timeout (seconds) for optimization mode")
    parser.add_argument("--early_stopping_rounds", type=int, default=200, help="Early stopping rounds for optimization mode")
    parser.add_argument("--seed", type=int, default=42, help="")
    args = parser.parse_args()
    main(args)
