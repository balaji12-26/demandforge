"""
DemandForge MPC Optimizer
=========================
Model Predictive Control dispatch using PuLP (CBC solver).
Minimizes total cost = energy cost + demand charge + degradation.
"""

import numpy as np
import pandas as pd

try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False

import config as cfg
from models import solar_generation, orc_generation, get_tariff


def run_mpc_optimization(
    load_profile: np.ndarray,
    orc_profile: np.ndarray = None,
    soc_initial: float = cfg.SOC_INITIAL,
) -> pd.DataFrame:
    """
    Solve the optimal battery dispatch problem over 96 time steps.

    Decision variables (per step t):
        P_charge[t]    : battery charging power (kW) >= 0
        P_discharge[t] : battery discharging power (kW) >= 0
        P_grid[t]      : grid import power (kW) >= 0

    Objective:
        Minimize sum_t [ P_grid[t] * tariff[t] * dt
                        + degradation_cost * (P_charge[t] + P_discharge[t]) * dt ]
                + demand_charge_penalty * P_grid_max

    Constraints:
        Power balance:  solar[t] + orc[t] + P_discharge[t] + P_grid[t]
                        = load[t] + P_charge[t]  (+ curtailment slack)
        SOC dynamics:   SOC[t+1] = SOC[t] + (P_charge[t]*eff_c - P_discharge[t]/eff_d)*dt / cap
        SOC limits:     SOC_MIN <= SOC[t] <= SOC_MAX
        Grid cap:       P_grid[t] <= 0.60 * load[t]
        Inverter:       P_charge[t] <= inverter_max; P_discharge[t] <= inverter_max
    """
    if not PULP_AVAILABLE:
        print("[MPC] PuLP not available. Returning empty DataFrame.")
        return pd.DataFrame()

    N = cfg.TOTAL_STEPS
    dt = cfg.DT_HOURS
    cap = cfg.BESS_NOMINAL_KWH

    # Pre-compute generation and tariff arrays
    solar = np.array([solar_generation(i * dt) for i in range(N)])
    if orc_profile is not None:
        orc = np.array(orc_profile, dtype=float)
    else:
        orc = np.array([orc_generation(i * dt) for i in range(N)])
    tariff = np.array([get_tariff(i * dt) for i in range(N)])

    # --- Build LP ---
    prob = pulp.LpProblem("DemandForge_MPC", pulp.LpMinimize)

    # Decision variables
    P_charge = [pulp.LpVariable(f"Pch_{t}", lowBound=0, upBound=cfg.MAX_CHARGE_RATE_KW) for t in range(N)]
    P_discharge = [pulp.LpVariable(f"Pdis_{t}", lowBound=0, upBound=cfg.MAX_DISCHARGE_RATE_KW) for t in range(N)]
    P_grid = [pulp.LpVariable(f"Pgrid_{t}", lowBound=0) for t in range(N)]
    P_curtail = [pulp.LpVariable(f"Pcurt_{t}", lowBound=0) for t in range(N)]
    SOC = [pulp.LpVariable(f"SOC_{t}", lowBound=cfg.SOC_MIN, upBound=cfg.SOC_MAX) for t in range(N + 1)]

    # Peak grid variable (for demand charge)
    P_grid_max = pulp.LpVariable("Pgrid_max", lowBound=0)

    # --- Objective ---
    # Daily demand charge contribution (monthly / 30)
    demand_charge_weight = cfg.DEMAND_CHARGE_PER_KVA / 0.95 / 30.0  # Rs per kW-day

    prob += (
        pulp.lpSum([P_grid[t] * tariff[t] * dt for t in range(N)])
        + pulp.lpSum([(P_charge[t] + P_discharge[t]) * cfg.DEGRADATION_COST_PER_KWH * dt for t in range(N)])
        + demand_charge_weight * P_grid_max
    )

    # --- Constraints ---
    # Initial SOC
    prob += SOC[0] == soc_initial, "SOC_init"

    for t in range(N):
        load_t = float(load_profile[t])
        solar_t = float(solar[t])
        orc_t = float(orc[t])

        # Power balance
        prob += (
            solar_t + orc_t + P_discharge[t] + P_grid[t]
            == load_t + P_charge[t] + P_curtail[t],
            f"balance_{t}",
        )

        # SOC dynamics
        prob += (
            SOC[t + 1] == SOC[t]
            + (P_charge[t] * cfg.CHARGE_EFFICIENCY - P_discharge[t] / cfg.DISCHARGE_EFFICIENCY)
            * dt / cap,
            f"soc_dyn_{t}",
        )

        # Grid cap
        prob += P_grid[t] <= cfg.GRID_CAP_FRACTION * load_t, f"grid_cap_{t}"

        # Peak grid tracking
        prob += P_grid_max >= P_grid[t], f"peak_grid_{t}"

    # --- Solve ---
    solver = pulp.PULP_CBC_CMD(msg=0, timeLimit=60)
    status = prob.solve(solver)

    if status != pulp.constants.LpStatusOptimal:
        print(f"[MPC] Solver status: {pulp.LpStatus[status]}. Trying with relaxed constraints...")
        # Return empty if infeasible
        return pd.DataFrame()

    # --- Extract results ---
    records = []
    for t in range(N):
        hour = t * dt
        bess_net = pulp.value(P_discharge[t]) - pulp.value(P_charge[t])
        records.append({
            "step": t,
            "time": f"{int(hour):02d}:{int((hour % 1) * 60):02d}",
            "hour": hour,
            "load_kw": float(load_profile[t]),
            "solar_kw": float(solar[t]),
            "orc_kw": float(orc[t]),
            "bess_kw": bess_net,
            "grid_kw": pulp.value(P_grid[t]),
            "soc_pct": pulp.value(SOC[t]) * 100.0,
            "tariff": float(tariff[t]),
            "curtailed_kw": pulp.value(P_curtail[t]),
            "bess_charge_kw": pulp.value(P_charge[t]),
            "bess_discharge_kw": pulp.value(P_discharge[t]),
        })

    df = pd.DataFrame(records)

    obj_val = pulp.value(prob.objective)
    print(f"[MPC] Optimal daily cost: Rs. {obj_val:,.2f}")
    print(f"[MPC] Peak grid import: {pulp.value(P_grid_max):.1f} kW")

    return df
