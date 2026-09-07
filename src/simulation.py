"""
DemandForge Simulation Engine
==============================
Rule-based dispatch: charge battery from solar surplus, discharge at peak,
grid fills the gap (capped at 60% of load).
"""

import numpy as np
import pandas as pd
import config as cfg
from models import solar_generation, orc_generation, generate_load_profile, BatteryModel, get_tariff


def run_simulation(
    load_profile: np.ndarray = None,
    orc_profile: np.ndarray = None,
    with_demand_side: bool = True,
    soc_initial: float = cfg.SOC_INITIAL,
    label: str = "rule_based",
) -> pd.DataFrame:
    """
    Run the 96-step rule-based dispatch simulation.

    Dispatch priority:
        1. Solar + ORC serve load directly
        2. Surplus solar charges battery
        3. During peak hours or when renewables < load, discharge battery
        4. Grid fills remaining gap (capped at 60% of load)
        5. If grid cap prevents full supply, discharge more battery

    Args:
        orc_profile: Optional array of ORC output per step (kW).
                     If provided, overrides orc_generation() for variable ORC.

    Returns DataFrame with columns:
        time, hour, load_kw, solar_kw, orc_kw, bess_kw, grid_kw,
        soc_pct, tariff, curtailed_kw
    """
    if load_profile is None:
        load_profile = generate_load_profile(with_demand_side=with_demand_side)

    battery = BatteryModel(soc_initial=soc_initial)

    records = []

    for step in range(cfg.TOTAL_STEPS):
        hour = step * cfg.DT_HOURS
        load = load_profile[step]

        # Generation
        solar = solar_generation(hour)
        orc = orc_profile[step] if orc_profile is not None else orc_generation(hour)
        renewable = solar + orc

        # Grid cap
        grid_max = cfg.GRID_CAP_FRACTION * load

        tariff = get_tariff(hour)
        is_peak = cfg.PEAK_START_HOUR <= hour < cfg.PEAK_END_HOUR

        # --- Dispatch Logic ---
        bess_kw = 0.0       # positive = discharge, negative = charge
        grid_kw = 0.0
        curtailed = 0.0

        if renewable >= load:
            # Surplus: charge battery with excess
            surplus = renewable - load
            accepted = battery.charge(surplus)
            bess_kw = -accepted  # negative = charging
            curtailed = surplus - accepted  # if battery full
            grid_kw = 0.0
        else:
            # Deficit: need battery and/or grid
            deficit = load - renewable

            if is_peak or (hour >= 17.0 and hour < 22.0):
                # Peak / evening: discharge battery first, grid fills gap
                discharge = battery.discharge(min(deficit, cfg.INVERTER_MAX_KW))
                bess_kw = discharge
                remaining = deficit - discharge
                grid_kw = min(remaining, grid_max)
                # If grid cap prevents full supply, try more battery
                shortfall = remaining - grid_kw
                if shortfall > 0:
                    extra = battery.discharge(shortfall)
                    bess_kw += extra
                    shortfall -= extra
                    if shortfall > 0:
                        # Last resort: allow grid to exceed cap slightly
                        grid_kw += shortfall
            elif hour >= 6.0 and hour < 17.0:
                # Daytime off-peak: charge battery if there's solar, grid fills rest
                # (renewable < load here, so no surplus to charge)
                # Use grid for deficit, try to preserve battery
                grid_kw = min(deficit, grid_max)
                remaining = deficit - grid_kw
                if remaining > 0:
                    discharge = battery.discharge(remaining)
                    bess_kw = discharge
                    remaining -= discharge
                    if remaining > 0:
                        grid_kw += remaining
            else:
                # Night: grid + battery if needed
                grid_kw = min(deficit, grid_max)
                remaining = deficit - grid_kw
                if remaining > 0:
                    discharge = battery.discharge(remaining)
                    bess_kw = discharge
                    remaining -= discharge
                    if remaining > 0:
                        grid_kw += remaining

        # Record
        soc_pct = battery.soc * 100.0
        records.append({
            "step": step,
            "time": f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
            "hour": hour,
            "load_kw": load,
            "solar_kw": solar,
            "orc_kw": orc,
            "bess_kw": bess_kw,       # +discharge, -charge
            "grid_kw": grid_kw,
            "soc_pct": soc_pct,
            "tariff": tariff,
            "curtailed_kw": curtailed,
        })

    df = pd.DataFrame(records)

    # Validate energy balance
    _validate_energy_balance(df)

    return df


def _validate_energy_balance(df: pd.DataFrame, tolerance: float = 1.0):
    """
    Verify that generation = consumption at every time step.
    solar + orc + bess_discharge + grid = load + bess_charge + curtailed
    """
    for _, row in df.iterrows():
        bess_discharge = max(row["bess_kw"], 0)
        bess_charge = max(-row["bess_kw"], 0)
        supply = row["solar_kw"] + row["orc_kw"] + bess_discharge + row["grid_kw"]
        demand = row["load_kw"] + bess_charge + row["curtailed_kw"]
        imbalance = abs(supply - demand)
        if imbalance > tolerance:
            print(f"  [WARN] Energy imbalance at step {row['step']}: "
                  f"supply={supply:.1f} kW, demand={demand:.1f} kW, "
                  f"delta={imbalance:.1f} kW")


def calculate_costs(df: pd.DataFrame) -> dict:
    """Calculate electricity costs from a simulation result DataFrame."""
    # Energy cost
    energy_cost = (df["grid_kw"] * cfg.DT_HOURS * df["tariff"]).sum()

    # Demand charge (based on maximum grid import in the month)
    # Approximate power factor = 0.95, so kVA = kW / 0.95
    max_grid_kw = df["grid_kw"].max()
    max_grid_kva = max_grid_kw / 0.95
    demand_charge = max_grid_kva * cfg.DEMAND_CHARGE_PER_KVA  # monthly

    # Battery degradation cost
    bess_throughput = df["bess_kw"].abs().sum() * cfg.DT_HOURS  # kWh cycled
    degradation_cost = bess_throughput * cfg.DEGRADATION_COST_PER_KWH

    # Total daily cost (demand charge pro-rated to 1 day = /30)
    total_daily = energy_cost + demand_charge / 30.0 + degradation_cost

    return {
        "energy_cost_rs": energy_cost,
        "demand_charge_monthly_rs": demand_charge,
        "demand_charge_daily_rs": demand_charge / 30.0,
        "degradation_cost_rs": degradation_cost,
        "total_daily_cost_rs": total_daily,
        "max_grid_kw": max_grid_kw,
        "max_grid_kva": max_grid_kva,
        "total_grid_kwh": (df["grid_kw"] * cfg.DT_HOURS).sum(),
        "total_solar_kwh": (df["solar_kw"] * cfg.DT_HOURS).sum(),
        "total_orc_kwh": (df["orc_kw"] * cfg.DT_HOURS).sum(),
        "total_load_kwh": (df["load_kw"] * cfg.DT_HOURS).sum(),
        "total_bess_discharge_kwh": (df["bess_kw"].clip(lower=0) * cfg.DT_HOURS).sum(),
        "total_bess_charge_kwh": ((-df["bess_kw"]).clip(lower=0) * cfg.DT_HOURS).sum(),
        "total_curtailed_kwh": (df["curtailed_kw"] * cfg.DT_HOURS).sum(),
    }


def run_baseline_simulation(load_profile: np.ndarray = None) -> pd.DataFrame:
    """
    Baseline scenario: NO solar, NO ORC, NO battery. Grid supplies everything.
    Used for cost comparison.
    """
    if load_profile is None:
        load_profile = generate_load_profile(with_demand_side=False)

    records = []
    for step in range(cfg.TOTAL_STEPS):
        hour = step * cfg.DT_HOURS
        load = load_profile[step]
        tariff = get_tariff(hour)
        records.append({
            "step": step,
            "time": f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
            "hour": hour,
            "load_kw": load,
            "solar_kw": 0.0,
            "orc_kw": 0.0,
            "bess_kw": 0.0,
            "grid_kw": load,  # grid supplies all
            "soc_pct": 0.0,
            "tariff": tariff,
            "curtailed_kw": 0.0,
        })
    return pd.DataFrame(records)
