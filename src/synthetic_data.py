"""
DemandForge Synthetic Data Generator
=====================================
Generates 30 days of 15-min resolution synthetic data for an AP
metal-processing cluster with continuous reheating furnace (~400 C exhaust).

Columns: Timestamp, Electrical_Demand_kW, Solar_Generation_kW,
         Furnace_Thermal_kWth, ORC_Electrical_kW, Ambient_C,
         Day_Type, Prev_Hour_Load_kW
"""

import numpy as np
import pandas as pd
import config as cfg

# ── Physical parameters ──
ORC_ELECTRICAL_EFFICIENCY = 0.18     # ORC thermal → electrical (400°C exhaust)
THERMAL_BASE_KWTH = 800.0            # standby / night thermal output
THERMAL_DAY_KWTH = 1800.0            # daytime production thermal
THERMAL_PEAK_KWTH = 2900.0           # peak production thermal (evening)
BATCH_RIPPLE_AMPLITUDE = 150.0       # kWth oscillation per batch cycle
BATCH_RIPPLE_PERIOD_STEPS = 3        # 3 steps = 45 min batch cycle

AMBIENT_BASE_C = 28.0                # nighttime ambient (AP coastal)
AMBIENT_PEAK_C = 38.0                # afternoon peak
AMBIENT_VARIATION_C = 2.0            # day-to-day variation


def generate_synthetic_dataset(
    n_days: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate n_days of 15-min resolution data.
    Returns DataFrame with 2880 rows (30 days × 96 steps).
    """
    rng = np.random.RandomState(seed)
    n_steps = n_days * cfg.TOTAL_STEPS
    records = []

    for day in range(n_days):
        # Day type: 0=weekday, 1=weekend (assume Mon-Fri work, Sat-Sun off)
        day_type = 1 if (day % 7) >= 5 else 0
        weekend_scale = 0.65 if day_type == 1 else 1.0

        # Day-to-day variation
        demand_scale = 1.0 + rng.uniform(-0.08, 0.08)
        thermal_scale = 1.0 + rng.uniform(-0.06, 0.06)
        cloud_factor = rng.uniform(0.80, 1.0)  # cloud cover reduces solar
        ambient_offset = rng.uniform(-AMBIENT_VARIATION_C, AMBIENT_VARIATION_C)

        prev_load = cfg.NIGHT_LOAD_KW  # initialize for first step

        for step in range(cfg.TOTAL_STEPS):
            hour = step * cfg.DT_HOURS
            global_step = day * cfg.TOTAL_STEPS + step
            ts = pd.Timestamp("2026-06-01") + pd.Timedelta(minutes=global_step * cfg.TIME_STEP_MINUTES)

            # ── 1. Electrical Demand ──
            load = _electrical_demand(hour, rng, demand_scale, weekend_scale)

            # ── 2. Solar Generation ──
            solar = _solar_generation(hour, cloud_factor)

            # ── 3. Furnace Thermal Output (with batch ripple) ──
            thermal = _furnace_thermal(
                hour, step, rng, thermal_scale, weekend_scale
            )

            # ── 4. ORC Electrical (derived from thermal, capped) ──
            orc = min(thermal * ORC_ELECTRICAL_EFFICIENCY, cfg.ORC_CAPACITY_KW)

            # ── 5. Ambient Temperature ──
            ambient = _ambient_temperature(hour, ambient_offset)

            records.append({
                "Timestamp": ts,
                "Day": day + 1,
                "Step": step,
                "Hour": hour,
                "Day_Type": day_type,
                "Electrical_Demand_kW": round(load, 1),
                "Solar_Generation_kW": round(solar, 1),
                "Furnace_Thermal_kWth": round(thermal, 1),
                "ORC_Electrical_kW": round(orc, 1),
                "Ambient_C": round(ambient, 1),
                "Prev_Hour_Load_kW": round(prev_load, 1),
            })

            prev_load = load

    df = pd.DataFrame(records)
    return df


def _electrical_demand(
    hour: float, rng: np.random.RandomState,
    demand_scale: float, weekend_scale: float,
) -> float:
    """Industrial load profile: night → ramp → day → evening surge → ramp down."""
    base = cfg.BASE_LOAD_KW
    peak = base * cfg.PEAK_FACTOR

    if hour < 6.0:
        load = cfg.NIGHT_LOAD_KW
    elif hour < 8.0:
        frac = (hour - 6.0) / 2.0
        load = cfg.NIGHT_LOAD_KW + frac * (base - cfg.NIGHT_LOAD_KW)
    elif hour < 17.0:
        load = base
    elif hour < 17.5:
        # Pre-surge ramp (gradual)
        frac = (hour - 17.0) / 0.5
        load = base + frac * (base * 0.15)  # slight pre-increase
    elif hour < 18.0:
        # Surge ramp 17:30-18:00
        frac = (hour - 17.5) / 0.5
        load = base * 1.15 + frac * (peak - base * 1.15)
    elif hour < 22.0:
        # Evening peak with motor inrush spikes
        spike = rng.uniform(0.0, 0.08) * peak
        load = peak + spike
    else:
        # Post-peak ramp down
        frac = (hour - 22.0) / 2.0
        load = peak - frac * (peak - cfg.EVENING_RAMPDOWN_KW)

    # Apply scaling and noise
    load *= demand_scale * weekend_scale
    load += rng.normal(0, 12)
    return max(load, 50.0)


def _solar_generation(hour: float, cloud_factor: float) -> float:
    """
    Solar parabola centered at 12.5, ZERO at 18:00.
    Uses parabolic curve: P = Pmax * (1 - ((h - 12.5) / 5.5)^2)
    """
    center = 12.5
    half_width = 5.5  # sunrise ~7:00, sunset exactly 18:00

    if hour < (center - half_width) or hour >= 18.0:
        return 0.0

    x = (hour - center) / half_width
    output = cfg.SOLAR_CAPACITY_KWP * cfg.SOLAR_EFFICIENCY * max(0, 1.0 - x * x)
    return output * cloud_factor


def _furnace_thermal(
    hour: float, step: int, rng: np.random.RandomState,
    thermal_scale: float, weekend_scale: float,
) -> float:
    """
    Furnace waste heat with production-driven variation + batch ripple.
    - Night: standby heat (800 kWth)
    - Day: production heat (1800 kWth)
    - Evening: heavy production ramp (2900 kWth) with batch cycling
    """
    if hour < 6.0:
        # Night standby
        thermal = THERMAL_BASE_KWTH
    elif hour < 8.0:
        # Morning startup
        frac = (hour - 6.0) / 2.0
        thermal = THERMAL_BASE_KWTH + frac * (THERMAL_DAY_KWTH - THERMAL_BASE_KWTH)
    elif hour < 16.0:
        # Daytime production
        thermal = THERMAL_DAY_KWTH
    elif hour < 18.0:
        # Evening ramp (16:00-18:00) — furnaces loading for heavy production
        frac = (hour - 16.0) / 2.0
        thermal = THERMAL_DAY_KWTH + frac * (THERMAL_PEAK_KWTH - THERMAL_DAY_KWTH)
    elif hour < 22.0:
        # Heavy production with batch ripple
        ripple = BATCH_RIPPLE_AMPLITUDE * np.sin(
            2 * np.pi * step / BATCH_RIPPLE_PERIOD_STEPS
        )
        thermal = THERMAL_PEAK_KWTH + ripple
    else:
        # Post-production rampdown
        frac = (hour - 22.0) / 2.0
        thermal = THERMAL_PEAK_KWTH - frac * (THERMAL_PEAK_KWTH - THERMAL_BASE_KWTH)

    # Apply scaling, weekend reduction, and noise
    thermal *= thermal_scale * weekend_scale
    thermal += rng.normal(0, 30)
    return max(thermal, 100.0)


def _ambient_temperature(hour: float, offset: float) -> float:
    """Diurnal temperature cycle for AP region."""
    # Sinusoidal: min at 5 AM, max at 15:00 (3 PM)
    phase = 2 * np.pi * (hour - 5.0) / 24.0
    temp = AMBIENT_BASE_C + (AMBIENT_PEAK_C - AMBIENT_BASE_C) * (
        0.5 * (1 + np.sin(phase - np.pi / 2))
    )
    return temp + offset


def get_representative_day(
    df: pd.DataFrame, day_num: int = None,
) -> tuple:
    """
    Extract load and ORC profiles for a specific day.
    Returns (load_profile, orc_profile) as numpy arrays of length 96.
    """
    if day_num is None:
        # Use the last weekday in the dataset
        weekdays = df[df["Day_Type"] == 0]["Day"].unique()
        day_num = weekdays[-1] if len(weekdays) > 0 else df["Day"].max()

    day_data = df[df["Day"] == day_num].sort_values("Step")
    load_profile = day_data["Electrical_Demand_kW"].values
    orc_profile = day_data["ORC_Electrical_kW"].values
    thermal_profile = day_data["Furnace_Thermal_kWth"].values

    return load_profile, orc_profile, thermal_profile


if __name__ == "__main__":
    print("Generating 30-day synthetic dataset...")
    df = generate_synthetic_dataset(n_days=30, seed=42)
    out_path = "synthetic_30day.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved to {out_path}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDay types: {df.groupby('Day_Type').size().to_dict()}")
    print(f"Load range: {df['Electrical_Demand_kW'].min():.0f} - {df['Electrical_Demand_kW'].max():.0f} kW")
    print(f"Thermal range: {df['Furnace_Thermal_kWth'].min():.0f} - {df['Furnace_Thermal_kWth'].max():.0f} kWth")
    print(f"ORC range: {df['ORC_Electrical_kW'].min():.0f} - {df['ORC_Electrical_kW'].max():.0f} kW")
    print(f"Solar max: {df['Solar_Generation_kW'].max():.0f} kW")
