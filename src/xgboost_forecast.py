"""
DemandForge XGBoost Co-Forecast
================================
Two-output XGBoost model that jointly predicts:
  1. Electrical_Demand_kW
  2. Furnace_Thermal_kWth

This is the AI/ML differentiator: forecasting BOTH electrical load
AND furnace thermal output lets the MPC optimizer know how much ORC
power will be available during the evening peak.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

import config as cfg


def prepare_features(df: pd.DataFrame) -> tuple:
    """
    Build feature matrix X and target matrix Y from the synthetic dataset.

    Features: Hour, Prev_Hour_Load_kW, Ambient_C, Day_Type,
              Hour_sin, Hour_cos (cyclical encoding)
    Targets:  Electrical_Demand_kW, Furnace_Thermal_kWth
    """
    X = pd.DataFrame({
        "Hour": df["Hour"],
        "Hour_sin": np.sin(2 * np.pi * df["Hour"] / 24.0),
        "Hour_cos": np.cos(2 * np.pi * df["Hour"] / 24.0),
        "Prev_Hour_Load_kW": df["Prev_Hour_Load_kW"],
        "Ambient_C": df["Ambient_C"],
        "Day_Type": df["Day_Type"],
    })

    Y = df[["Electrical_Demand_kW", "Furnace_Thermal_kWth"]].values

    return X, Y


def train_xgboost_model(
    df: pd.DataFrame,
    test_days: int = 5,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> dict:
    """
    Train a multi-output XGBoost model.

    Split: last `test_days` days as test set, rest as training.
    Returns dict with model, predictions, metrics, and data splits.
    """
    total_days = df["Day"].max()
    train_cutoff = total_days - test_days

    # Split by day (temporal split, not random)
    train_df = df[df["Day"] <= train_cutoff]
    test_df = df[df["Day"] > train_cutoff]

    X_train, Y_train = prepare_features(train_df)
    X_test, Y_test = prepare_features(test_df)

    print(f"[XGBoost] Training on {len(X_train)} samples ({train_cutoff} days)")
    print(f"[XGBoost] Testing on  {len(X_test)} samples ({test_days} days)")

    # Build multi-output XGBoost
    base_model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=seed,
        verbosity=0,
        tree_method="hist",
    )
    model = MultiOutputRegressor(base_model)
    model.fit(X_train, Y_train)

    # Predictions
    Y_pred_train = model.predict(X_train)
    Y_pred_test = model.predict(X_test)

    # Metrics for each target
    target_names = ["Electrical_Demand_kW", "Furnace_Thermal_kWth"]
    metrics = {}

    for i, name in enumerate(target_names):
        actual_test = Y_test[:, i]
        pred_test = Y_pred_test[:, i]
        actual_train = Y_train[:, i]
        pred_train = Y_pred_train[:, i]

        mae_test = mean_absolute_error(actual_test, pred_test)
        rmse_test = np.sqrt(mean_squared_error(actual_test, pred_test))
        r2_test = r2_score(actual_test, pred_test)

        # MAPE (avoiding division by zero)
        mask = actual_test > 10
        mape_test = np.mean(np.abs(
            (actual_test[mask] - pred_test[mask]) / actual_test[mask]
        )) * 100

        mae_train = mean_absolute_error(actual_train, pred_train)

        metrics[name] = {
            "mae_test": mae_test,
            "rmse_test": rmse_test,
            "r2_test": r2_test,
            "mape_test": mape_test,
            "mae_train": mae_train,
        }

        print(f"[XGBoost] {name}:")
        print(f"  Train MAE: {mae_train:.1f} | Test MAE: {mae_test:.1f}")
        print(f"  Test RMSE: {rmse_test:.1f} | Test MAPE: {mape_test:.1f}%")
        print(f"  Test R²:   {r2_test:.4f}")

    # Feature importance (average across both outputs)
    importances = {}
    feature_names = list(X_train.columns)
    for est in model.estimators_:
        fi = est.feature_importances_
        for j, fname in enumerate(feature_names):
            importances[fname] = importances.get(fname, 0) + fi[j] / len(model.estimators_)

    # Sort by importance
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    print("\n[XGBoost] Feature Importance (averaged):")
    for fname, imp in importances.items():
        print(f"  {fname:25s} {imp:.4f}")

    return {
        "model": model,
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "Y_pred_train": Y_pred_train,
        "Y_pred_test": Y_pred_test,
        "metrics": metrics,
        "feature_importance": importances,
        "target_names": target_names,
        "test_df": test_df,
        "train_df": train_df,
    }


def get_forecast_for_day(
    model,
    df: pd.DataFrame,
    day_num: int,
) -> tuple:
    """
    Use the trained model to forecast a specific day.
    Returns (predicted_load, predicted_thermal) as numpy arrays of length 96.
    """
    day_data = df[df["Day"] == day_num].sort_values("Step")
    X, _ = prepare_features(day_data)
    Y_pred = model.predict(X)

    return Y_pred[:, 0], Y_pred[:, 1]  # load, thermal


if __name__ == "__main__":
    from synthetic_data import generate_synthetic_dataset

    print("Generating synthetic data...")
    df = generate_synthetic_dataset(n_days=30, seed=42)
    print(f"Dataset shape: {df.shape}")

    print("\nTraining XGBoost co-forecast model...")
    results = train_xgboost_model(df, test_days=5)
    print("\nDone!")
