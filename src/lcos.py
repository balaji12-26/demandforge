"""
DemandForge LCOS Calculator
=============================
Levelized Cost of Storage (LCOS) analysis with full CAPEX/OPEX breakdown,
battery degradation schedule, NPV, IRR, and payback period.

References:
- APERC HT Industry tariff schedule (2025-26)
- IRENA Battery Storage Cost Survey 2023
- NITI Aayog EV/Storage roadmap (India-specific)
"""

import numpy as np
import config as cfg


# ── CAPEX Breakdown (Rs.) ──
CAPEX_SOLAR_PER_KWP = 42000        # Rs/kWp installed (incl. mounting, DC cabling)
CAPEX_ORC_PER_KW = 85000           # Rs/kW ORC system (400 C waste heat)
CAPEX_BESS_PER_KWH = 11500         # Rs/kWh LFP cell + BMS + thermal management
CAPEX_INVERTER_PER_KW = 4500       # Rs/kW grid-forming inverter
CAPEX_BOS_FRACTION = 0.12          # Balance of System as fraction of equipment cost

# ── OPEX ──
OPEX_SOLAR_PER_KWP_YEAR = 600      # Rs/kWp/year (cleaning, monitoring)
OPEX_ORC_PER_KW_YEAR = 3500        # Rs/kW/year (filters, lube, minor overhaul)
OPEX_BESS_PER_KWH_YEAR = 200       # Rs/kWh/year (monitoring, cooling)
OPEX_GENERAL_ANNUAL = 300000       # Rs/year (insurance, admin, grid fees)

# ── Battery Degradation ──
BESS_CYCLE_LIFE = 6000             # cycles to 80% capacity (LFP)
BESS_CALENDAR_LIFE_YEARS = 15      # calendar life
BESS_REPLACEMENT_YEAR = 10         # replacement year
BESS_REPLACEMENT_COST_FRACTION = 0.55  # replacement at 55% of original (costs decline)
ANNUAL_DEGRADATION_PCT = 2.5       # % capacity loss per year (linear approx)

# ── Financial ──
PROJECT_LIFETIME_YEARS = 25
DISCOUNT_RATE = 0.10               # 10% WACC
DEBT_FRACTION = 0.70               # 70:30 debt-equity
INTEREST_RATE = 0.10               # 10% debt interest
TARIFF_ESCALATION = 0.03           # 3% annual tariff increase
GRID_TARIFF_RS_KWH = cfg.BASE_TARIFF_PER_KWH


def calculate_capex() -> dict:
    """Calculate total CAPEX breakdown."""
    solar = CAPEX_SOLAR_PER_KWP * cfg.SOLAR_CAPACITY_KWP
    orc = CAPEX_ORC_PER_KW * cfg.ORC_CAPACITY_KW
    bess = CAPEX_BESS_PER_KWH * cfg.BESS_NOMINAL_KWH
    inverter = CAPEX_INVERTER_PER_KW * cfg.INVERTER_MAX_KW
    equipment_total = solar + orc + bess + inverter
    bos = equipment_total * CAPEX_BOS_FRACTION
    total = equipment_total + bos

    return {
        "solar_rs": solar,
        "orc_rs": orc,
        "bess_rs": bess,
        "inverter_rs": inverter,
        "bos_rs": bos,
        "total_rs": total,
        "total_cr": total / 1e7,  # in Crores
    }


def calculate_annual_opex() -> dict:
    """Calculate annual OPEX breakdown."""
    solar = OPEX_SOLAR_PER_KWP_YEAR * cfg.SOLAR_CAPACITY_KWP
    orc = OPEX_ORC_PER_KW_YEAR * cfg.ORC_CAPACITY_KW
    bess = OPEX_BESS_PER_KWH_YEAR * cfg.BESS_NOMINAL_KWH
    general = OPEX_GENERAL_ANNUAL
    total = solar + orc + bess + general

    return {
        "solar_rs": solar,
        "orc_rs": orc,
        "bess_rs": bess,
        "general_rs": general,
        "total_rs": total,
    }


def calculate_annual_savings(costs_baseline: dict, costs_optimized: dict) -> float:
    """Calculate annual savings from the HRES vs grid-only baseline."""
    daily_saving = costs_baseline["total_daily_cost_rs"] - costs_optimized["total_daily_cost_rs"]
    # Assume 300 operating days/year (industrial facility)
    return daily_saving * 300


def calculate_lcos(
    costs_baseline: dict,
    costs_optimized: dict,
    annual_bess_throughput_kwh: float = None,
) -> dict:
    """
    Full LCOS calculation.

    LCOS = Total discounted lifecycle costs / Total discounted energy throughput

    Returns comprehensive financial analysis.
    """
    capex = calculate_capex()
    opex = calculate_annual_opex()
    annual_savings = calculate_annual_savings(costs_baseline, costs_optimized)

    # Annual energy served by the HRES (solar + ORC + BESS discharge)
    # This is the correct denominator for LCOS of the entire system
    if annual_bess_throughput_kwh is None:
        daily_renewable = (
            costs_optimized.get("total_solar_kwh", 0) +
            costs_optimized.get("total_orc_kwh", 0) +
            costs_optimized.get("total_bess_discharge_kwh", 0)
        )
        annual_bess_throughput_kwh = daily_renewable * 300  # 300 operating days

    # ── Year-by-year cash flow ──
    years = list(range(PROJECT_LIFETIME_YEARS + 1))
    cash_flows = []
    discounted_costs = []
    discounted_savings = []
    discounted_throughput = []
    cumulative_cash = []

    running_sum = 0

    for year in years:
        discount = (1 + DISCOUNT_RATE) ** year

        if year == 0:
            # Year 0: CAPEX (outflow)
            cost = capex["total_rs"]
            saving = 0
            throughput = 0
        else:
            # Annual OPEX
            cost = opex["total_rs"]

            # Battery replacement
            if year == BESS_REPLACEMENT_YEAR:
                cost += capex["bess_rs"] * BESS_REPLACEMENT_COST_FRACTION

            # Degradation: slightly reduce throughput each year
            degradation_factor = max(0.7, 1.0 - ANNUAL_DEGRADATION_PCT / 100 * year)

            # Tariff escalation increases savings over time
            tariff_factor = (1 + TARIFF_ESCALATION) ** year
            saving = annual_savings * tariff_factor
            throughput = annual_bess_throughput_kwh * degradation_factor

        net_cash = saving - cost
        running_sum += net_cash

        cash_flows.append({
            "year": year,
            "cost_rs": cost,
            "saving_rs": saving,
            "net_cash_rs": net_cash,
            "cumulative_rs": running_sum,
            "discounted_cost": cost / discount,
            "discounted_saving": saving / discount,
            "throughput_kwh": throughput,
            "discounted_throughput": throughput / discount,
        })

        discounted_costs.append(cost / discount)
        discounted_savings.append(saving / discount)
        discounted_throughput.append(throughput / discount)
        cumulative_cash.append(running_sum)

    # ── LCOS ──
    total_discounted_cost = sum(discounted_costs)
    total_discounted_throughput = sum(discounted_throughput)
    lcos_rs_kwh = total_discounted_cost / total_discounted_throughput if total_discounted_throughput > 0 else float('inf')

    # ── NPV ──
    npv = sum(cf["discounted_saving"] - cf["discounted_cost"] for cf in cash_flows)

    # ── Simple Payback ──
    payback_year = None
    for cf in cash_flows:
        if cf["cumulative_rs"] > 0:
            payback_year = cf["year"]
            break

    # ── IRR (Newton-Raphson approximation) ──
    net_flows = [-capex["total_rs"]] + [
        cash_flows[y]["saving_rs"] - cash_flows[y]["cost_rs"]
        for y in range(1, len(cash_flows))
    ]
    irr = _calculate_irr(net_flows)

    # ── Avoided grid cost comparison ──
    annual_grid_cost_baseline = costs_baseline["total_daily_cost_rs"] * 300
    annual_grid_cost_hres = costs_optimized["total_daily_cost_rs"] * 300

    print(f"\n[LCOS] -- Financial Summary --")
    print(f"  CAPEX:           Rs. {capex['total_rs']:>12,.0f} ({capex['total_cr']:.2f} Cr)")
    print(f"  Annual OPEX:     Rs. {opex['total_rs']:>12,.0f}")
    print(f"  Annual Savings:  Rs. {annual_savings:>12,.0f}")
    print(f"  LCOS:            Rs. {lcos_rs_kwh:>12.2f} /kWh")
    print(f"  NPV (25 yr):     Rs. {npv:>12,.0f}")
    print(f"  IRR:             {irr*100:>12.1f} %")
    print(f"  Payback:         {payback_year:>12d} years")

    return {
        "capex": capex,
        "opex": opex,
        "annual_savings_rs": annual_savings,
        "lcos_rs_kwh": lcos_rs_kwh,
        "npv_rs": npv,
        "irr": irr,
        "payback_years": payback_year,
        "cash_flows": cash_flows,
        "annual_grid_baseline_rs": annual_grid_cost_baseline,
        "annual_grid_hres_rs": annual_grid_cost_hres,
    }


def _calculate_irr(cash_flows: list, max_iter: int = 200, tol: float = 1e-6) -> float:
    """Calculate IRR using Newton-Raphson method."""
    rate = 0.10  # initial guess

    for _ in range(max_iter):
        npv = sum(cf / (1 + rate) ** t for t, cf in enumerate(cash_flows))
        dnpv = sum(-t * cf / (1 + rate) ** (t + 1) for t, cf in enumerate(cash_flows))

        if abs(dnpv) < 1e-12:
            break

        rate -= npv / dnpv

        if abs(npv) < tol:
            break

    return max(0, rate)


if __name__ == "__main__":
    # Quick test with mock data
    mock_baseline = {
        "total_daily_cost_rs": 204692.12,
        "total_bess_discharge_kwh": 0,
    }
    mock_optimized = {
        "total_daily_cost_rs": 73141.89,
        "total_bess_discharge_kwh": 1753.7,
    }

    result = calculate_lcos(mock_baseline, mock_optimized)
    print(f"\nLCOS: Rs. {result['lcos_rs_kwh']:.2f}/kWh")
    print(f"Grid tariff: Rs. {cfg.BASE_TARIFF_PER_KWH:.2f}/kWh")
    print(f"LCOS < Grid tariff: {result['lcos_rs_kwh'] < cfg.BASE_TARIFF_PER_KWH}")
