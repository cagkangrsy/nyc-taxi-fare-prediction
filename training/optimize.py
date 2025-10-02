import gc
import json
import random
import warnings
from pathlib import Path
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.visualization.matplotlib import plot_optimization_history
from sklearn.metrics import root_mean_squared_error

warnings.filterwarnings("ignore")


def optimize_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    artifact_root: str,
    n_trials: int = 50,
    timeout: int | None = None,
    seed: int = 42,
    early_stopping_rounds: int = 200,
):
    """Run Optuna optimization for LightGBM and save artifacts.

    Returns a tuple: (best_value_rmse, best_params, exp_dir)
    """

    np.random.seed(seed)
    random.seed(seed)

    exp_dir = Path(artifact_root) / "model_experiments" / "optuna_lgbm"
    exp_dir.mkdir(parents=True, exist_ok=True)

    study_name = "lgbm_regression_optimization"
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", study_name=study_name, sampler=sampler)

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 500, 3000, step=250),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 511, step=32),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": seed,
            "n_jobs": 1,
            "feature_fraction_seed": seed,
            "bagging_seed": seed,
            "data_random_seed": seed,
            "verbosity": -1,
        }

        model = lgb.LGBMRegressor(**params)
        callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False)]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="rmse",
            callbacks=callbacks,
        )

        preds = model.predict(X_val)
        score = root_mean_squared_error(y_val, preds)
        trial.set_user_attr("valid_rmse", score)

        del model
        gc.collect()

        return score

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=1,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    (exp_dir / "best_params.json").write_text(json.dumps(study.best_params, indent=2))

    def trial_to_dict(t: optuna.trial.FrozenTrial):
        return {
            "number": t.number,
            "value": t.value,
            "state": str(t.state),
            "params": t.params,
            "user_attrs": t.user_attrs,
            "duration": t.duration.total_seconds() if t.duration else None,
            "datetime_start": t.datetime_start.isoformat() if t.datetime_start else None,
            "datetime_complete": (
                t.datetime_complete.isoformat() if t.datetime_complete else None
            ),
        }

    with (exp_dir / "trials.json").open("w", encoding="utf-8") as f:
        json.dump([trial_to_dict(t) for t in study.trials], f, indent=2)

    plt.figure()
    plot_optimization_history(study)
    plt.tight_layout()
    plt.savefig(exp_dir / "optimization_history.png", dpi=150)
    plt.close()

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

    return study.best_value, study.best_params, exp_dir