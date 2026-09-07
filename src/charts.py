"""
DemandForge Charts
==================
Professional matplotlib charts for the Hybrid Hack 2026 presentation.
All charts saved as 300 DPI PNGs to the charts/ subdirectory.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import config as cfg

# Ensure charts directory exists
CHART_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), cfg.CHART_DIR)
os.makedirs(CHART_PATH, exist_ok=True)


def _setup_style():
    """Set consistent professional chart style."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "axes.grid.which": "major",
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.dpi": 100,
        "savefig.dpi": cfg.CHART_DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.15,
    })


_setup_style()


def _hours_axis(ax, label="Time of Day"):
    """Configure x-axis for 24-hour time display."""
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 25, 2)], rotation=45, ha="right")
    ax.set_xlabel(label)


def _add_peak_shading(ax):
    """Add light shading for peak hours (6-10 PM)."""
    ax.axvspan(cfg.PEAK_START_HOUR, cfg.PEAK_END_HOUR, alpha=0.08, color="red", label="Peak Hours (6-10 PM)")


def _save(fig, name):
    """Save figure and close."""
    path = os.path.join(CHART_PATH, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  [Chart] Saved: {path}")


# ============================================================
# Chart 1: Load Profile Before/After DemandForge
# ============================================================
def chart_load_profile(load_before: np.ndarray, load_after: np.ndarray):
    hours = np.arange(cfg.TOTAL_STEPS) * cfg.DT_HOURS

    fig, ax = plt.subplots(figsize=(12, 5))
    _add_peak_shading(ax)
    ax.plot(hours, load_before, color="#D32F2F", linewidth=2, label="Without DemandForge", alpha=0.8)
    ax.plot(hours, load_after, color=cfg.COLOR_DSM, linewidth=2, label="With DemandForge (DSM)")
    ax.fill_between(hours, load_before, load_after,
                     where=load_before > load_after, alpha=0.2, color=cfg.COLOR_DSM,
                     label="Peak Reduction")
    ax.set_ylabel("Load (kW)")
    ax.set_title("Industrial Load Profile: Before vs After DemandForge")
    _hours_axis(ax)
    ax.legend(loc="upper left")
    _save(fig, "01_load_profile.png")


# ============================================================
# Chart 2: Energy Mix Stacked Area
# ============================================================
def chart_energy_mix(df: pd.DataFrame):
    hours = df["hour"].values

    solar = df["solar_kw"].values
    orc = df["orc_kw"].values
    bess_discharge = np.maximum(df["bess_kw"].values, 0)
    grid = df["grid_kw"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    _add_peak_shading(ax)

    ax.stackplot(
        hours, solar, orc, bess_discharge, grid,
        labels=["Solar PV", "ORC Waste-to-Energy", "Battery Discharge", "Grid Import"],
        colors=[cfg.COLOR_SOLAR, cfg.COLOR_ORC, cfg.COLOR_BATTERY, cfg.COLOR_GRID],
        alpha=0.85,
    )
    ax.plot(hours, df["load_kw"].values, color=cfg.COLOR_LOAD, linewidth=2,
            linestyle="--", label="Load Demand")

    ax.set_ylabel("Power (kW)")
    ax.set_title("Energy Mix: 24-Hour Dispatch Profile")
    _hours_axis(ax)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    _save(fig, "02_energy_mix.png")


# ============================================================
# Chart 3: Battery SOC Over 24 Hours
# ============================================================
def chart_battery_soc(df: pd.DataFrame, df_mpc: pd.DataFrame = None):
    hours = df["hour"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    _add_peak_shading(ax)

    ax.plot(hours, df["soc_pct"].values, color=cfg.COLOR_SOC, linewidth=2.5, label="Rule-Based SOC")
    if df_mpc is not None and len(df_mpc) > 0:
        ax.plot(df_mpc["hour"].values, df_mpc["soc_pct"].values, color="#7B1FA2",
                linewidth=2, linestyle="--", label="MPC-Optimized SOC")

    ax.axhline(y=cfg.SOC_MIN * 100, color="red", linestyle=":", linewidth=1.5, label=f"SOC Min ({cfg.SOC_MIN*100:.0f}%)")
    ax.axhline(y=cfg.SOC_MAX * 100, color="green", linestyle=":", linewidth=1.5, label=f"SOC Max ({cfg.SOC_MAX*100:.0f}%)")

    ax.fill_between(hours, cfg.SOC_MIN * 100, df["soc_pct"].values,
                     alpha=0.15, color=cfg.COLOR_SOC)

    ax.set_ylabel("State of Charge (%)")
    ax.set_title("Battery SOC Profile Over 24 Hours")
    ax.set_ylim(0, 100)
    _hours_axis(ax)
    ax.legend(loc="upper right")
    _save(fig, "03_battery_soc.png")


# ============================================================
# Chart 4: Grid Import vs 60% Cap
# ============================================================
def chart_grid_cap(df: pd.DataFrame):
    hours = df["hour"].values
    grid = df["grid_kw"].values
    cap = cfg.GRID_CAP_FRACTION * df["load_kw"].values

    fig, ax = plt.subplots(figsize=(12, 5))
    _add_peak_shading(ax)

    ax.fill_between(hours, grid, alpha=0.4, color=cfg.COLOR_GRID, label="Actual Grid Import")
    ax.plot(hours, grid, color=cfg.COLOR_GRID, linewidth=2)
    ax.plot(hours, cap, color="#D32F2F", linewidth=2, linestyle="--", label="60% Cap Limit")
    ax.plot(hours, df["load_kw"].values, color=cfg.COLOR_LOAD, linewidth=1.5,
            linestyle=":", label="Total Load", alpha=0.6)

    ax.set_ylabel("Power (kW)")
    ax.set_title("Grid Import vs 60% Cap Constraint")
    _hours_axis(ax)
    ax.legend(loc="upper left")
    ax.set_ylim(bottom=0)
    _save(fig, "04_grid_cap.png")


# ============================================================
# Chart 5: Cost Comparison Bar Chart
# ============================================================
def chart_cost_comparison(costs_baseline: dict, costs_demandforge: dict, costs_mpc: dict = None):
    categories = ["Energy Cost\n(Daily)", "Demand Charge\n(Daily Share)", "Total Daily\nCost"]
    baseline_vals = [
        costs_baseline["energy_cost_rs"],
        costs_baseline["demand_charge_daily_rs"],
        costs_baseline["total_daily_cost_rs"],
    ]
    df_vals = [
        costs_demandforge["energy_cost_rs"],
        costs_demandforge["demand_charge_daily_rs"],
        costs_demandforge["total_daily_cost_rs"],
    ]

    x = np.arange(len(categories))
    width = 0.28

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - width, baseline_vals, width, label="Baseline (Grid Only)",
                    color="#D32F2F", alpha=0.85)
    bars2 = ax.bar(x, df_vals, width, label="DemandForge (Rule-Based)",
                    color=cfg.COLOR_DSM, alpha=0.85)

    if costs_mpc is not None:
        mpc_vals = [
            costs_mpc["energy_cost_rs"],
            costs_mpc["demand_charge_daily_rs"],
            costs_mpc["total_daily_cost_rs"],
        ]
        bars3 = ax.bar(x + width, mpc_vals, width, label="DemandForge (MPC)",
                        color="#7B1FA2", alpha=0.85)

    ax.set_ylabel("Cost (Rs.)")
    ax.set_title("Daily Cost Comparison: Baseline vs DemandForge")
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()

    # Add value labels on bars
    for bar_group in [bars1, bars2]:
        for bar in bar_group:
            height = bar.get_height()
            ax.annotate(f"Rs.{height:,.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    _save(fig, "05_cost_comparison.png")


# ============================================================
# Chart 6: Demand Charge Comparison
# ============================================================
def chart_demand_charge(costs_baseline: dict, costs_demandforge: dict, costs_mpc: dict = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Peak grid kW
    labels = ["Baseline", "Rule-Based"]
    peak_kw = [costs_baseline["max_grid_kw"], costs_demandforge["max_grid_kw"]]
    colors = ["#D32F2F", cfg.COLOR_DSM]
    if costs_mpc is not None:
        labels.append("MPC")
        peak_kw.append(costs_mpc["max_grid_kw"])
        colors.append("#7B1FA2")

    bars1 = ax1.bar(labels, peak_kw, color=colors, alpha=0.85)
    ax1.set_ylabel("Peak Grid Import (kW)")
    ax1.set_title("Recorded Maximum Demand")
    for bar, val in zip(bars1, peak_kw):
        ax1.annotate(f"{val:.0f} kW", xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

    # Right: Monthly demand charge
    demand_charges = [costs_baseline["demand_charge_monthly_rs"],
                      costs_demandforge["demand_charge_monthly_rs"]]
    if costs_mpc is not None:
        demand_charges.append(costs_mpc["demand_charge_monthly_rs"])

    bars2 = ax2.bar(labels, demand_charges, color=colors, alpha=0.85)
    ax2.set_ylabel("Monthly Demand Charge (Rs.)")
    ax2.set_title("Demand Charge Impact")
    for bar, val in zip(bars2, demand_charges):
        ax2.annotate(f"Rs.{val:,.0f}", xy=(bar.get_x() + bar.get_width() / 2, val),
                     xytext=(0, 4), textcoords="offset points", ha="center", fontsize=10)

    fig.suptitle("Demand Charge Reduction Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "06_demand_charge.png")


# ============================================================
# Chart 7: MPC vs Rule-Based Dispatch
# ============================================================
def chart_mpc_comparison(df_rule: pd.DataFrame, df_mpc: pd.DataFrame):
    if df_mpc is None or len(df_mpc) == 0:
        print("  [Chart] Skipping MPC comparison (no MPC data)")
        return

    hours = df_rule["hour"].values

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Grid import comparison
    ax = axes[0]
    _add_peak_shading(ax)
    ax.plot(hours, df_rule["grid_kw"].values, color=cfg.COLOR_GRID, linewidth=2, label="Rule-Based Grid")
    ax.plot(hours, df_mpc["grid_kw"].values, color="#7B1FA2", linewidth=2, linestyle="--", label="MPC Grid")
    ax.set_ylabel("Grid (kW)")
    ax.set_title("MPC vs Rule-Based: Grid Import")
    ax.legend(loc="upper right")

    # Battery dispatch comparison
    ax = axes[1]
    _add_peak_shading(ax)
    ax.plot(hours, df_rule["bess_kw"].values, color=cfg.COLOR_BATTERY, linewidth=2, label="Rule-Based BESS")
    ax.plot(hours, df_mpc["bess_kw"].values, color="#7B1FA2", linewidth=2, linestyle="--", label="MPC BESS")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("BESS Power (kW)\n(+discharge / -charge)")
    ax.set_title("MPC vs Rule-Based: Battery Dispatch")
    ax.legend(loc="upper right")

    # SOC comparison
    ax = axes[2]
    _add_peak_shading(ax)
    ax.plot(hours, df_rule["soc_pct"].values, color=cfg.COLOR_SOC, linewidth=2, label="Rule-Based SOC")
    ax.plot(hours, df_mpc["soc_pct"].values, color="#7B1FA2", linewidth=2, linestyle="--", label="MPC SOC")
    ax.axhline(y=cfg.SOC_MIN * 100, color="red", linestyle=":", alpha=0.7)
    ax.axhline(y=cfg.SOC_MAX * 100, color="green", linestyle=":", alpha=0.7)
    ax.set_ylabel("SOC (%)")
    ax.set_ylim(0, 100)
    ax.set_title("MPC vs Rule-Based: Battery SOC")
    _hours_axis(ax)
    ax.legend(loc="upper right")

    fig.tight_layout()
    _save(fig, "07_mpc_comparison.png")


# ============================================================
# Chart 8: Forecast vs Actual Load
# ============================================================
def chart_forecast(forecast_results: dict):
    hours = np.arange(cfg.TOTAL_STEPS) * cfg.DT_HOURS
    actual = forecast_results["actual"]
    sma = forecast_results["sma_forecast"]
    wma = forecast_results["wma_forecast"]
    upper = forecast_results["upper_band"]
    lower = forecast_results["lower_band"]

    fig, ax = plt.subplots(figsize=(12, 5))
    _add_peak_shading(ax)

    ax.fill_between(hours, lower, upper, alpha=0.15, color=cfg.COLOR_BATTERY, label="95% Confidence Band")
    ax.plot(hours, actual, color=cfg.COLOR_LOAD, linewidth=2, label="Actual Load")
    ax.plot(hours, sma, color="#E65100", linewidth=2, linestyle="--", label="SMA Forecast (7-day)")
    ax.plot(hours, wma, color="#7B1FA2", linewidth=1.5, linestyle=":", label="WMA Forecast (7-day)")

    sma_m = forecast_results["sma_metrics"]
    wma_m = forecast_results["wma_metrics"]
    textstr = (f"SMA: MAPE={sma_m['mape_pct']:.1f}%, RMSE={sma_m['rmse_kw']:.0f} kW\n"
               f"WMA: MAPE={wma_m['mape_pct']:.1f}%, RMSE={wma_m['rmse_kw']:.0f} kW")
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    ax.set_ylabel("Load (kW)")
    ax.set_title("Load Forecasting: Prediction vs Actual")
    _hours_axis(ax)
    ax.legend(loc="upper right")
    _save(fig, "08_forecast.png")


# ============================================================
# Chart 9: Energy Balance Pie Chart
# ============================================================
def chart_energy_pie(costs: dict):
    # Generation side
    labels = []
    sizes = []
    colors = []

    if costs["total_solar_kwh"] > 0:
        labels.append(f"Solar PV\n{costs['total_solar_kwh']:.0f} kWh")
        sizes.append(costs["total_solar_kwh"])
        colors.append(cfg.COLOR_SOLAR)
    if costs["total_orc_kwh"] > 0:
        labels.append(f"ORC W2E\n{costs['total_orc_kwh']:.0f} kWh")
        sizes.append(costs["total_orc_kwh"])
        colors.append(cfg.COLOR_ORC)
    if costs["total_grid_kwh"] > 0:
        labels.append(f"Grid Import\n{costs['total_grid_kwh']:.0f} kWh")
        sizes.append(costs["total_grid_kwh"])
        colors.append(cfg.COLOR_GRID)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    wedges1, texts1, autotexts1 = ax1.pie(
        sizes, labels=labels, colors=colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    ax1.set_title("Energy Generation Sources", fontweight="bold")

    # Consumption side
    load_kwh = costs["total_load_kwh"]
    bess_charge_kwh = costs["total_bess_charge_kwh"]
    curtailed_kwh = costs.get("total_curtailed_kwh", 0)

    labels2 = [f"Load Served\n{load_kwh:.0f} kWh"]
    sizes2 = [load_kwh]
    colors2 = [cfg.COLOR_LOAD]
    if bess_charge_kwh > 0:
        labels2.append(f"BESS Charging\n{bess_charge_kwh:.0f} kWh")
        sizes2.append(bess_charge_kwh)
        colors2.append(cfg.COLOR_BATTERY)
    if curtailed_kwh > 1:
        labels2.append(f"Curtailed\n{curtailed_kwh:.0f} kWh")
        sizes2.append(curtailed_kwh)
        colors2.append(cfg.COLOR_EXCESS)

    wedges2, texts2, autotexts2 = ax2.pie(
        sizes2, labels=labels2, colors=colors2, autopct="%1.1f%%",
        startangle=140, pctdistance=0.75,
        textprops={"fontsize": 9},
    )
    ax2.set_title("Energy Consumption Breakdown", fontweight="bold")

    fig.suptitle("24-Hour Energy Balance", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "09_energy_pie.png")


# ============================================================
# Chart 10: Efficiency Waterfall / Breakdown
# ============================================================
def chart_efficiency_breakdown(costs: dict, costs_baseline: dict):
    """Horizontal bar chart showing efficiency improvements."""
    total_load = costs["total_load_kwh"]
    total_gen = costs["total_solar_kwh"] + costs["total_orc_kwh"] + costs["total_grid_kwh"]

    solar_pct = costs["total_solar_kwh"] / total_gen * 100
    orc_pct = costs["total_orc_kwh"] / total_gen * 100
    grid_pct = costs["total_grid_kwh"] / total_gen * 100

    baseline_grid = costs_baseline["total_grid_kwh"]
    grid_reduction = (1 - costs["total_grid_kwh"] / baseline_grid) * 100

    peak_reduction = (1 - costs["max_grid_kw"] / costs_baseline["max_grid_kw"]) * 100

    cost_saving = (1 - costs["total_daily_cost_rs"] / costs_baseline["total_daily_cost_rs"]) * 100

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Energy Mix breakdown
    ax = axes[0]
    categories = ["Solar PV", "ORC W2E", "Grid"]
    values = [solar_pct, orc_pct, grid_pct]
    colors = [cfg.COLOR_SOLAR, cfg.COLOR_ORC, cfg.COLOR_GRID]
    bars = ax.barh(categories, values, color=colors, alpha=0.85, height=0.5)
    ax.set_xlabel("Share of Total Generation (%)")
    ax.set_title("Generation Mix")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, max(values) * 1.3)

    # Reduction metrics
    ax = axes[1]
    metrics = ["Grid Energy\nReduction", "Peak Demand\nReduction", "Cost\nSaving"]
    vals = [grid_reduction, peak_reduction, cost_saving]
    bar_colors = [cfg.COLOR_DSM if v > 0 else "#D32F2F" for v in vals]
    bars = ax.barh(metrics, vals, color=bar_colors, alpha=0.85, height=0.5)
    ax.set_xlabel("Reduction (%)")
    ax.set_title("DemandForge Impact")
    ax.axvline(x=40, color="red", linestyle="--", alpha=0.5, label="40% Target")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10, fontweight="bold")
    ax.legend(loc="lower right")
    ax.set_xlim(0, max(max(vals) * 1.3, 50))

    # Round-trip efficiency
    ax = axes[2]
    # Use design RTE (the discharge/charge ratio is misleading when initial SOC != final SOC)
    rt_eff = cfg.ROUNDTRIP_EFFICIENCY * 100  # Design round-trip efficiency

    renewable_gen = costs.get("total_solar_kwh", 0) + costs.get("total_orc_kwh", 0)
    renewable_frac = (renewable_gen / total_gen * 100) if total_gen > 0 else 0

    eff_items = ["BESS Round-Trip", "Solar Derating", "Renewable\nFraction"]
    eff_vals = [rt_eff, cfg.SOLAR_EFFICIENCY * 100, renewable_frac]
    bars = ax.barh(eff_items, eff_vals, color=[cfg.COLOR_BATTERY, cfg.COLOR_SOLAR, cfg.COLOR_DSM],
                   alpha=0.85, height=0.5)
    ax.set_xlabel("Efficiency / Fraction (%)")
    ax.set_title("System Efficiency")
    for bar, val in zip(bars, eff_vals):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va="center", fontsize=10)
    ax.set_xlim(0, 110)

    fig.suptitle("DemandForge System Performance", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "10_efficiency_breakdown.png")


# ===================================================================
# NEW CHARTS: Thermal/ORC, XGBoost, LCOS
# ===================================================================

def chart_thermal_orc(synthetic_df, representative_day=None):
    """Chart 11: Furnace thermal + ORC electrical for representative day."""
    _setup_style()

    if representative_day is None:
        weekdays = synthetic_df[synthetic_df["Day_Type"] == 0]["Day"].unique()
        representative_day = weekdays[-1] if len(weekdays) > 0 else synthetic_df["Day"].max()

    day_data = synthetic_df[synthetic_df["Day"] == representative_day].sort_values("Step")
    hours = day_data["Hour"].values
    thermal = day_data["Furnace_Thermal_kWth"].values
    orc_elec = day_data["ORC_Electrical_kW"].values

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Left axis: Thermal
    color_thermal = "#D84315"
    ax1.fill_between(hours, thermal, alpha=0.25, color=color_thermal)
    ax1.plot(hours, thermal, color=color_thermal, linewidth=2.0, label="Furnace Thermal (kWth)")
    ax1.set_ylabel("Furnace Waste Heat (kWth)", color=color_thermal)
    ax1.tick_params(axis="y", labelcolor=color_thermal)
    ax1.set_ylim(0, max(thermal) * 1.15)

    # Right axis: ORC electrical
    ax2 = ax1.twinx()
    color_orc = cfg.COLOR_ORC
    ax2.plot(hours, orc_elec, color=color_orc, linewidth=2.5, linestyle="-", label="ORC Electrical (kW)")
    ax2.axhline(y=cfg.ORC_CAPACITY_KW, color="red", linestyle="--", alpha=0.6, linewidth=1.5, label=f"ORC Cap ({cfg.ORC_CAPACITY_KW:.0f} kW)")
    ax2.set_ylabel("ORC Electrical Output (kW)", color=color_orc)
    ax2.tick_params(axis="y", labelcolor=color_orc)
    ax2.set_ylim(0, cfg.ORC_CAPACITY_KW * 1.3)

    _hours_axis(ax1)
    _add_peak_shading(ax1)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

    ax1.set_title(f"Furnace Thermal Output & ORC Generation (Day {representative_day})")
    fig.tight_layout()
    _save(fig, "11_thermal_orc.png")


def chart_xgboost_forecast(xgb_results):
    """Chart 12: XGBoost co-forecast — actual vs predicted for both targets."""
    _setup_style()

    Y_test = xgb_results["Y_test"]
    Y_pred = xgb_results["Y_pred_test"]
    test_df = xgb_results["test_df"]
    metrics = xgb_results["metrics"]

    # Get hours for x-axis (flatten all test days)
    hours = test_df["Hour"].values
    days = test_df["Day"].values
    n_per_day = cfg.TOTAL_STEPS

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    target_labels = ["Electrical Demand (kW)", "Furnace Thermal (kWth)"]
    target_keys = ["Electrical_Demand_kW", "Furnace_Thermal_kWth"]
    colors_actual = [cfg.COLOR_LOAD, "#D84315"]
    colors_pred = [cfg.COLOR_DSM, cfg.COLOR_ORC]

    for i, ax in enumerate(axes):
        actual = Y_test[:, i]
        predicted = Y_pred[:, i]
        x_vals = np.arange(len(actual))

        ax.plot(x_vals, actual, color=colors_actual[i], linewidth=1.5, alpha=0.8, label="Actual")
        ax.plot(x_vals, predicted, color=colors_pred[i], linewidth=1.5, linestyle="--", alpha=0.9, label="XGBoost Predicted")

        # Shade error
        ax.fill_between(x_vals, actual, predicted, alpha=0.15, color=colors_pred[i])

        m = metrics[target_keys[i]]
        ax.set_ylabel(target_labels[i])
        ax.legend(loc="upper right", fontsize=9)

        # Metrics annotation
        ax.text(0.02, 0.95, f"MAE: {m['mae_test']:.1f}  |  RMSE: {m['rmse_test']:.1f}  |  R²: {m['r2_test']:.4f}",
                transform=ax.transAxes, fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        # Add day separators
        unique_days = sorted(test_df["Day"].unique())
        for d_idx, d in enumerate(unique_days):
            sep = d_idx * n_per_day
            if sep > 0:
                ax.axvline(x=sep, color="gray", linestyle=":", alpha=0.4)

    # X-axis labels (show day boundaries)
    unique_days = sorted(test_df["Day"].unique())
    tick_pos = [i * n_per_day + n_per_day // 2 for i in range(len(unique_days))]
    axes[1].set_xticks(tick_pos)
    axes[1].set_xticklabels([f"Day {d}" for d in unique_days])
    axes[1].set_xlabel("Test Days")

    fig.suptitle("XGBoost Co-Forecast: Electrical Demand & Furnace Thermal", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "12_xgboost_forecast.png")


def chart_xgboost_importance(xgb_results):
    """Chart 13: Feature importance bar chart."""
    _setup_style()

    importance = xgb_results["feature_importance"]
    features = list(importance.keys())
    values = list(importance.values())

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#1565C0", "#FFB300", "#E65100", "#2E7D32", "#AB47BC", "#0288D1"]
    bars = ax.barh(features[::-1], values[::-1], color=colors[:len(features)], alpha=0.85, height=0.5)

    for bar, val in zip(bars, values[::-1]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10)

    ax.set_xlabel("Average Importance Score")
    ax.set_title("XGBoost Feature Importance (Averaged Over Both Targets)")
    ax.set_xlim(0, max(values) * 1.2)
    fig.tight_layout()
    _save(fig, "13_xgboost_importance.png")


def chart_lcos(lcos_results):
    """Chart 14: LCOS — CAPEX breakdown + cumulative cash flow with payback."""
    _setup_style()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Left panel: CAPEX breakdown ---
    ax = axes[0]
    capex = lcos_results["capex"]
    components = ["Solar PV\n1 MWp", "ORC\n500 kW", "BESS\n3.5 MWh", "Inverter\n1.5 MW", "BOS &\nInstall"]
    values_cr = [
        capex["solar_rs"] / 1e7,
        capex["orc_rs"] / 1e7,
        capex["bess_rs"] / 1e7,
        capex["inverter_rs"] / 1e7,
        capex["bos_rs"] / 1e7,
    ]
    colors = [cfg.COLOR_SOLAR, cfg.COLOR_ORC, cfg.COLOR_BATTERY, "#9E9E9E", cfg.COLOR_DSM]
    bars = ax.bar(components, values_cr, color=colors, alpha=0.85, width=0.6)

    for bar, val in zip(bars, values_cr):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"Rs.{val:.2f} Cr", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_ylabel("Cost (Rs. Crores)")
    ax.set_title(f"CAPEX Breakdown\nTotal: Rs. {capex['total_cr']:.2f} Crores")
    ax.set_ylim(0, max(values_cr) * 1.35)

    # --- Right panel: Cumulative cash flow ---
    ax = axes[1]
    cash_flows = lcos_results["cash_flows"]
    years = [cf["year"] for cf in cash_flows]
    cumulative = [cf["cumulative_rs"] / 1e7 for cf in cash_flows]  # in Crores

    ax.fill_between(years, cumulative, alpha=0.2, color=cfg.COLOR_DSM)
    ax.plot(years, cumulative, color=cfg.COLOR_DSM, linewidth=2.5, marker="o", markersize=3)
    ax.axhline(y=0, color="red", linestyle="--", alpha=0.6, linewidth=1.5)

    payback = lcos_results["payback_years"]
    if payback is not None:
        ax.axvline(x=payback, color="blue", linestyle="--", alpha=0.5)
        ax.annotate(f"Payback: Year {payback}",
                    xy=(payback, 0), xytext=(payback + 2, min(cumulative) * 0.4),
                    fontsize=10, fontweight="bold", color="blue",
                    arrowprops=dict(arrowstyle="->", color="blue"))

    ax.set_xlabel("Year")
    ax.set_ylabel("Cumulative Cash Flow (Rs. Crores)")
    ax.set_title(f"Financial Payback\nLCOS: Rs. {lcos_results['lcos_rs_kwh']:.2f}/kWh | IRR: {lcos_results['irr']*100:.1f}%")

    fig.suptitle("DemandForge Economic Analysis", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, "14_lcos_analysis.png")


def generate_all_charts(
    load_before, load_after, df_rule, df_mpc,
    costs_baseline, costs_rule, costs_mpc,
    forecast_results,
    synthetic_df=None, xgb_results=None, lcos_results=None,
    representative_day=None,
):
    """Generate all charts (10 original + 4 new)."""
    print("\n=== Generating Charts ===")

    # Original 10 charts
    chart_load_profile(load_before, load_after)
    chart_energy_mix(df_rule)
    chart_battery_soc(df_rule, df_mpc)
    chart_grid_cap(df_rule)
    chart_cost_comparison(costs_baseline, costs_rule, costs_mpc)
    chart_demand_charge(costs_baseline, costs_rule, costs_mpc)
    chart_mpc_comparison(df_rule, df_mpc)
    chart_forecast(forecast_results)
    chart_energy_pie(costs_rule)
    chart_efficiency_breakdown(costs_rule, costs_baseline)

    # New charts (only if data is available)
    if synthetic_df is not None:
        chart_thermal_orc(synthetic_df, representative_day)

    if xgb_results is not None:
        chart_xgboost_forecast(xgb_results)
        chart_xgboost_importance(xgb_results)

    if lcos_results is not None:
        chart_lcos(lcos_results)

    print(f"=== All charts saved to {CHART_PATH} ===\n")

