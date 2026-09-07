"""
DemandForge Forecasting
========================
Synthetic load history generation, moving-average forecasting,
and error metrics (MAPE, RMSE).
"""

import numpy as np
import pandas as pd
import config as cfg
from models import generate_load_profile


def generate_historical_data(
    n_days: int = cfg.FORECAST_HISTORY_DAYS,
    variation: float = cfg.FORECAST_DAY_VARIATION,
    base_seed: int = 100,
) -> np.ndarray:
    """
    Generate n_days of synthetic 96-step load profiles.
    Each day has random scaling and slight pattern shifts.

    Returns shape (n_days, 96).
    """
    rng = np.random.RandomState(base_seed)
    history = np.zeros((n_days, cfg.TOTAL_STEPS))

    for d in range(n_days):
        # Vary peak factor and base load slightly each day
        scale = 1.0 + rng.uniform(-variation, variation)
        peak_shift = rng.uniform(-0.05, 0.05)

        profile = generate_load_profile(
            base_kw=cfg.BASE_LOAD_KW * scale,
            peak_factor=cfg.PEAK_FACTOR + peak_shift,
            with_demand_side=False,
            seed=base_seed + d * 7,
        )
        history[d] = profile

    return history


def moving_average_forecast(
    history: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    Forecast the next day's load using a simple moving average
    of the last `window` days.

    Returns a 96-element forecast array.
    """
    if history.shape[0] < window:
        window = history.shape[0]

    forecast = history[-window:].mean(axis=0)
    return forecast


def weighted_moving_average_forecast(
    history: np.ndarray,
    window: int = 7,
) -> np.ndarray:
    """
    Weighted moving average: more recent days get higher weight.
    """
    if history.shape[0] < window:
        window = history.shape[0]

    weights = np.arange(1, window + 1, dtype=float)
    weights /= weights.sum()

    recent = history[-window:]
    forecast = np.zeros(cfg.TOTAL_STEPS)
    for i, w in enumerate(weights):
        forecast += w * recent[i]

    return forecast


def calculate_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Calculate MAPE and RMSE between actual and predicted load profiles."""
    # Avoid division by zero
    mask = actual > 10.0  # ignore very small loads
    errors = actual[mask] - predicted[mask]

    rmse = np.sqrt(np.mean(errors ** 2))
    mape = np.mean(np.abs(errors) / actual[mask]) * 100.0
    mae = np.mean(np.abs(errors))
    max_error = np.max(np.abs(errors))

    return {
        "rmse_kw": rmse,
        "mape_pct": mape,
        "mae_kw": mae,
        "max_error_kw": max_error,
    }


def run_forecasting() -> dict:
    """
    Full forecasting pipeline:
    1. Generate 30 days of synthetic history
    2. Use day 31 as the 'actual' day
    3. Forecast using moving average and weighted MA
    4. Calculate metrics

    Returns dict with history, actual, forecasts, and metrics.
    """
    # Generate 31 days (30 history + 1 actual)
    all_data = generate_historical_data(n_days=31, base_seed=100)
    history = all_data[:30]
    actual = all_data[30]

    # Simple moving average
    sma_forecast = moving_average_forecast(history, window=7)
    sma_metrics = calculate_metrics(actual, sma_forecast)

    # Weighted moving average
    wma_forecast = weighted_moving_average_forecast(history, window=7)
    wma_metrics = calculate_metrics(actual, wma_forecast)

    # Error bands (using historical std dev of last 7 days)
    recent_std = history[-7:].std(axis=0)
    upper_band = sma_forecast + 1.96 * recent_std
    lower_band = sma_forecast - 1.96 * recent_std

    print(f"[Forecasting] SMA - MAPE: {sma_metrics['mape_pct']:.1f}%, RMSE: {sma_metrics['rmse_kw']:.1f} kW")
    print(f"[Forecasting] WMA - MAPE: {wma_metrics['mape_pct']:.1f}%, RMSE: {wma_metrics['rmse_kw']:.1f} kW")

    return {
        "history": history,
        "actual": actual,
        "sma_forecast": sma_forecast,
        "wma_forecast": wma_forecast,
        "sma_metrics": sma_metrics,
        "wma_metrics": wma_metrics,
        "upper_band": upper_band,
        "lower_band": lower_band,
    }
