"""
DemandForge Main Orchestrator
==============================
Complete HRES simulation pipeline:
1. Generate 30-day synthetic dataset (AP metal-processing cluster)
2. Train XGBoost co-forecast (electrical + thermal)
3. Extract representative day → variable ORC profile
4. Run baseline simulation (grid only)
5. Apply DSM + run rule-based dispatch with variable ORC
6. Run MPC-optimized dispatch with variable ORC
7. Run load forecasting (SMA/WMA for comparison)
8. Calculate LCOS economic analysis
9. Generate all 14 charts
10. Print comprehensive summary report
"""

import sys
import os
import numpy as np
import pandas as pd

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config as cfg
from models import generate_load_profile, apply_demand_side_management
from simulation import run_simulation, run_baseline_simulation, calculate_costs
from mpc_optimizer import run_mpc_optimization
from forecasting import run_forecasting
from synthetic_data import generate_synthetic_dataset, get_representative_day
from xgboost_forecast import train_xgboost_model
from lcos import calculate_lcos
from charts import generate_all_charts


def print_separator(title: str = ""):
    print("\n" + "=" * 70)
    if title:
        print(f"  {title}")
        print("=" * 70)


def print_costs(label: str, costs: dict):
    print(f"\n  --- {label} ---")
    print(f"  Total Load:          {costs['total_load_kwh']:>10,.1f} kWh")
    print(f"  Solar Generation:    {costs['total_solar_kwh']:>10,.1f} kWh")
    print(f"  ORC Generation:      {costs['total_orc_kwh']:>10,.1f} kWh")
    print(f"  Grid Import:         {costs['total_grid_kwh']:>10,.1f} kWh")
    print(f"  BESS Discharge:      {costs['total_bess_discharge_kwh']:>10,.1f} kWh")
    print(f"  BESS Charge:         {costs['total_bess_charge_kwh']:>10,.1f} kWh")
    print(f"  Curtailed:           {costs['total_curtailed_kwh']:>10,.1f} kWh")
    print(f"  Peak Grid Import:    {costs['max_grid_kw']:>10,.1f} kW")
    print(f"  Energy Cost:       Rs.{costs['energy_cost_rs']:>10,.2f}")
    print(f"  Demand Charge/mo:  Rs.{costs['demand_charge_monthly_rs']:>10,.2f}")
    print(f"  Degradation Cost:  Rs.{costs['degradation_cost_rs']:>10,.2f}")
    print(f"  Total Daily Cost:  Rs.{costs['total_daily_cost_rs']:>10,.2f}")


def main():
    print_separator("DemandForge HRES Simulation")
    print("  Team CityCoders | Hybrid Hack 2026, Round 2")
    print("  Simulation: 24 hours, 15-min steps (96 steps)")
    print(f"  Solar: {cfg.SOLAR_CAPACITY_KWP} kWp | ORC: {cfg.ORC_CAPACITY_KW} kW")
    print(f"  BESS: {cfg.BESS_NOMINAL_MWH} MWh ({cfg.USABLE_CAPACITY_KWH:.0f} kWh usable)")
    print(f"  Inverter: {cfg.INVERTER_MAX_KW} kW | Grid Cap: {cfg.GRID_CAP_FRACTION*100:.0f}%")

    # --------------------------------------------------------
    # Step 1: Generate 30-Day Synthetic Dataset
    # --------------------------------------------------------
    print_separator("Step 1: 30-Day Synthetic Data Generation")
    synthetic_df = generate_synthetic_dataset(n_days=30, seed=42)
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "synthetic_30day.csv")
    synthetic_df.to_csv(csv_path, index=False)

    n_weekdays = (synthetic_df.groupby("Day").first()["Day_Type"] == 0).sum()
    n_weekends = 30 - n_weekdays
    print(f"  Dataset: {len(synthetic_df)} rows ({30} days x {cfg.TOTAL_STEPS} steps)")
    print(f"  Weekdays: {n_weekdays} | Weekends: {n_weekends}")
    print(f"  Load range: {synthetic_df['Electrical_Demand_kW'].min():.0f} - {synthetic_df['Electrical_Demand_kW'].max():.0f} kW")
    print(f"  Thermal range: {synthetic_df['Furnace_Thermal_kWth'].min():.0f} - {synthetic_df['Furnace_Thermal_kWth'].max():.0f} kWth")
    print(f"  ORC range: {synthetic_df['ORC_Electrical_kW'].min():.0f} - {synthetic_df['ORC_Electrical_kW'].max():.0f} kW")
    print(f"  Solar max: {synthetic_df['Solar_Generation_kW'].max():.0f} kW")
    print(f"  Saved to: {csv_path}")

    # --------------------------------------------------------
    # Step 2: Train XGBoost Co-Forecast
    # --------------------------------------------------------
    print_separator("Step 2: XGBoost Co-Forecast Training")
    xgb_results = train_xgboost_model(synthetic_df, test_days=5)

    # --------------------------------------------------------
    # Step 3: Extract Representative Day
    # --------------------------------------------------------
    print_separator("Step 3: Representative Day Extraction")
    # Use the last weekday from the test set (most realistic for demo)
    test_days_df = xgb_results["test_df"]
    test_weekdays = test_days_df[test_days_df["Day_Type"] == 0]["Day"].unique()
    if len(test_weekdays) > 0:
        rep_day = test_weekdays[-1]  # last test weekday
    else:
        rep_day = test_days_df["Day"].max()

    load_raw, orc_profile, thermal_profile = get_representative_day(synthetic_df, rep_day)

    # Apply DSM to the raw load
    load_after = apply_demand_side_management(load_raw, seed=42)
    # Use raw load as the "before" (no DSM)
    load_before = load_raw.copy()

    print(f"  Representative day: Day {rep_day}")
    print(f"  Raw peak:          {load_before.max():>8.1f} kW")
    print(f"  DSM-managed peak:  {load_after.max():>8.1f} kW")
    print(f"  Peak reduction:    {(1 - load_after.max()/load_before.max())*100:>8.1f}%")
    print(f"  ORC range:         {orc_profile.min():.0f} - {orc_profile.max():.0f} kW")
    print(f"  Thermal range:     {thermal_profile.min():.0f} - {thermal_profile.max():.0f} kWth")

    # --------------------------------------------------------
    # Step 4: Baseline Simulation (Grid Only)
    # --------------------------------------------------------
    print_separator("Step 4: Baseline Simulation (Grid Only)")
    df_baseline = run_baseline_simulation(load_before)
    costs_baseline = calculate_costs(df_baseline)
    print_costs("Baseline (No HRES)", costs_baseline)

    # --------------------------------------------------------
    # Step 5: Rule-Based Dispatch with Variable ORC
    # --------------------------------------------------------
    print_separator("Step 5: Rule-Based Dispatch (Variable ORC)")
    df_rule = run_simulation(
        load_profile=load_after,
        orc_profile=orc_profile,
        with_demand_side=True,
    )
    costs_rule = calculate_costs(df_rule)
    print_costs("DemandForge (Rule-Based)", costs_rule)

    # SOC validation
    soc_min_actual = df_rule["soc_pct"].min()
    soc_max_actual = df_rule["soc_pct"].max()
    print(f"\n  SOC Range: {soc_min_actual:.1f}% - {soc_max_actual:.1f}%")
    assert soc_min_actual >= cfg.SOC_MIN * 100 - 0.1, f"SOC below minimum! ({soc_min_actual:.1f}%)"
    assert soc_max_actual <= cfg.SOC_MAX * 100 + 0.1, f"SOC above maximum! ({soc_max_actual:.1f}%)"
    print("  SOC constraints: PASSED")

    # --------------------------------------------------------
    # Step 6: MPC Optimization with Variable ORC
    # --------------------------------------------------------
    print_separator("Step 6: MPC-Optimized Dispatch (Variable ORC)")
    df_mpc = run_mpc_optimization(load_after, orc_profile=orc_profile)
    costs_mpc = None
    if df_mpc is not None and len(df_mpc) > 0:
        costs_mpc = calculate_costs(df_mpc)
        print_costs("DemandForge (MPC)", costs_mpc)
    else:
        print("  MPC optimization did not produce results.")

    # --------------------------------------------------------
    # Step 7: Load Forecasting (SMA/WMA baseline comparison)
    # --------------------------------------------------------
    print_separator("Step 7: Load Forecasting (SMA/WMA)")
    forecast_results = run_forecasting()

    # --------------------------------------------------------
    # Step 8: LCOS Economic Analysis
    # --------------------------------------------------------
    print_separator("Step 8: LCOS Economic Analysis")
    lcos_results = calculate_lcos(costs_baseline, costs_rule)

    # --------------------------------------------------------
    # Step 9: Key Performance Metrics
    # --------------------------------------------------------
    print_separator("Step 9: Performance Summary")

    # Grid reduction during peak hours (6-10 PM)
    peak_mask = (df_rule["hour"] >= cfg.PEAK_START_HOUR) & (df_rule["hour"] < cfg.PEAK_END_HOUR)
    baseline_peak_grid = df_baseline.loc[
        (df_baseline["hour"] >= cfg.PEAK_START_HOUR) & (df_baseline["hour"] < cfg.PEAK_END_HOUR),
        "grid_kw"
    ].sum()
    rule_peak_grid = df_rule.loc[peak_mask, "grid_kw"].sum()
    peak_grid_reduction = (1 - rule_peak_grid / baseline_peak_grid) * 100

    # Overall grid reduction
    overall_grid_reduction = (1 - costs_rule["total_grid_kwh"] / costs_baseline["total_grid_kwh"]) * 100

    # Cost savings
    cost_saving_pct = (1 - costs_rule["total_daily_cost_rs"] / costs_baseline["total_daily_cost_rs"]) * 100
    cost_saving_rs = costs_baseline["total_daily_cost_rs"] - costs_rule["total_daily_cost_rs"]
    monthly_saving = cost_saving_rs * 30

    # Renewable fraction
    total_gen = costs_rule["total_solar_kwh"] + costs_rule["total_orc_kwh"] + costs_rule["total_grid_kwh"]
    renewable_fraction = (costs_rule["total_solar_kwh"] + costs_rule["total_orc_kwh"]) / total_gen * 100

    print(f"\n  +-----------------------------------------------+----------+")
    print(f"  | Metric                                        |   Value  |")
    print(f"  +-----------------------------------------------+----------+")
    print(f"  | Peak-hour grid reduction (6-10 PM)            | {peak_grid_reduction:>6.1f}%  |")
    print(f"  | Overall grid energy reduction (24h)           | {overall_grid_reduction:>6.1f}%  |")
    print(f"  | Peak demand reduction (kW)                    | {(1-costs_rule['max_grid_kw']/costs_baseline['max_grid_kw'])*100:>6.1f}%  |")
    print(f"  | Renewable energy fraction                     | {renewable_fraction:>6.1f}%  |")
    print(f"  | Daily cost saving                             | {cost_saving_pct:>6.1f}%  |")
    print(f"  | Daily cost saving                             | Rs.{cost_saving_rs:>5.0f} |")
    print(f"  | Projected monthly saving                      | Rs.{monthly_saving:>5.0f} |")
    print(f"  | Battery SOC range                             | {soc_min_actual:.0f}-{soc_max_actual:.0f}%  |")
    print(f"  | BESS round-trip efficiency                    | {cfg.ROUNDTRIP_EFFICIENCY*100:>6.1f}%  |")
    print(f"  | LCOS                                          | Rs.{lcos_results['lcos_rs_kwh']:>4.2f}/kWh|")
    print(f"  | Grid tariff                                   | Rs.{cfg.BASE_TARIFF_PER_KWH:>4.2f}/kWh|")
    print(f"  | LCOS < Grid tariff                            | {'YES' if lcos_results['lcos_rs_kwh'] < cfg.BASE_TARIFF_PER_KWH else 'NO':>8s} |")
    print(f"  | IRR                                           | {lcos_results['irr']*100:>6.1f}%  |")
    print(f"  | Payback period                                | {lcos_results['payback_years']:>5d} yr  |")
    print(f"  +-----------------------------------------------+----------+")

    if peak_grid_reduction >= 40:
        print("\n  >>> 40% GRID REDUCTION TARGET: ACHIEVED <<<")
    else:
        print(f"\n  >>> Grid reduction at {peak_grid_reduction:.1f}% (target: 40%) <<<")

    # XGBoost metrics summary
    print("\n  --- XGBoost Co-Forecast Accuracy ---")
    for name, m in xgb_results["metrics"].items():
        short_name = name.replace("_kW", "").replace("_kWth", "")
        print(f"  {short_name:30s} MAE: {m['mae_test']:>6.1f} | R²: {m['r2_test']:.4f}")

    # --------------------------------------------------------
    # Step 10: Generate All Charts
    # --------------------------------------------------------
    generate_all_charts(
        load_before=load_before,
        load_after=load_after,
        df_rule=df_rule,
        df_mpc=df_mpc,
        costs_baseline=costs_baseline,
        costs_rule=costs_rule,
        costs_mpc=costs_mpc,
        forecast_results=forecast_results,
        synthetic_df=synthetic_df,
        xgb_results=xgb_results,
        lcos_results=lcos_results,
        representative_day=rep_day,
    )

    # --------------------------------------------------------
    # Step 11: Save Results to CSV
    # --------------------------------------------------------
    print_separator("Step 11: Saving Results")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    df_rule.to_csv(os.path.join(results_dir, "rule_based_dispatch.csv"), index=False)
    if df_mpc is not None and len(df_mpc) > 0:
        df_mpc.to_csv(os.path.join(results_dir, "mpc_dispatch.csv"), index=False)
    df_baseline.to_csv(os.path.join(results_dir, "baseline.csv"), index=False)

    print(f"  Results saved to {results_dir}")

    print_separator("DemandForge Simulation Complete")
    print("  All files generated successfully!")
    print(f"  Charts directory: {os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.CHART_DIR)}")
    print(f"  Results directory: {results_dir}")
    print(f"  Synthetic data: {csv_path}")
    print()

    return {
        "df_rule": df_rule,
        "df_mpc": df_mpc,
        "df_baseline": df_baseline,
        "costs_rule": costs_rule,
        "costs_mpc": costs_mpc,
        "costs_baseline": costs_baseline,
        "forecast_results": forecast_results,
        "synthetic_df": synthetic_df,
        "xgb_results": xgb_results,
        "lcos_results": lcos_results,
    }


if __name__ == "__main__":
    main()
