"""
DemandForge Configuration
=========================
All system parameters for the Hybrid Renewable Energy System simulation.
Team: CityCoders | Hybrid Hack 2026, Round 2
"""

# ---------- Simulation Parameters ----------
TIME_STEP_MINUTES = 15          # minutes per step
STEPS_PER_HOUR = 60 // TIME_STEP_MINUTES  # 4
TOTAL_STEPS = 96                # 24 hours * 4 steps/hour
DT_HOURS = TIME_STEP_MINUTES / 60.0  # 0.25 hours

# ---------- Load Parameters ----------
BASE_LOAD_KW = 1000.0           # daytime baseline (kW)
PEAK_FACTOR = 1.42              # peak/base ratio (~1350-1500 kW range)
NIGHT_LOAD_KW = 400.0           # night shift minimal load
EVENING_RAMPDOWN_KW = 600.0     # post-peak ramp-down target

# ---------- Solar PV ----------
SOLAR_CAPACITY_KWP = 1000.0     # 1.0 MWp
SOLAR_SUNRISE_HOUR = 6.0        # generation starts
SOLAR_SUNSET_HOUR = 17.5        # generation ends (5:30 PM)
SOLAR_PEAK_HOUR = 12.0          # solar noon
SOLAR_EFFICIENCY = 0.85         # derating (dust, temp, inverter losses)

# ---------- ORC Waste-to-Energy ----------
ORC_CAPACITY_KW = 500.0         # constant output during factory hours
ORC_FACTORY_START = 6.0         # factory starts at 6 AM
ORC_FACTORY_END = 22.0          # factory ends at 10 PM
ORC_NIGHT_FRACTION = 0.15       # 15% output at night (residual heat)

# ---------- Battery Energy Storage (BESS) ----------
BESS_NOMINAL_MWH = 3.5          # nominal capacity
BESS_NOMINAL_KWH = BESS_NOMINAL_MWH * 1000  # 3500 kWh
SOC_MIN = 0.20                  # minimum SOC (20%)
SOC_MAX = 0.95                  # maximum SOC (95%)
SOC_INITIAL = 0.50              # starting SOC
USABLE_CAPACITY_KWH = BESS_NOMINAL_KWH * (SOC_MAX - SOC_MIN)  # 2625 kWh
CHARGE_EFFICIENCY = 0.95        # one-way charge efficiency
DISCHARGE_EFFICIENCY = 0.95     # one-way discharge efficiency
ROUNDTRIP_EFFICIENCY = CHARGE_EFFICIENCY * DISCHARGE_EFFICIENCY  # 0.9025
INVERTER_MAX_KW = 1500.0        # 1.5 MW inverter limit
MAX_CHARGE_RATE_KW = INVERTER_MAX_KW   # limited by inverter
MAX_DISCHARGE_RATE_KW = INVERTER_MAX_KW

# ---------- Grid Parameters ----------
GRID_CAP_FRACTION = 0.60        # max 60% of load from grid
GRID_REDUCTION_TARGET = 0.40    # 40% reduction target

# ---------- Tariff Structure ----------
BASE_TARIFF_PER_KWH = 7.65      # Rs./kWh
TOD_SURCHARGE = 0.25            # 25% surcharge during peak
PEAK_START_HOUR = 18.0          # 6 PM
PEAK_END_HOUR = 22.0            # 10 PM
DEMAND_CHARGE_PER_KVA = 475.0   # Rs./kVA/month

# ---------- Demand-Side Management ----------
DSM_PEAK_REDUCTION = 0.17       # motor staggering reduces peak by 17%
DSM_LOAD_SHIFT_FRACTION = 0.08  # thermal pre-loading shifts 8% of evening kWh
DSM_SCHEDULING_REDUCTION = 0.05 # pump/compressor scheduling

# ---------- Battery Degradation ----------
DEGRADATION_COST_PER_KWH = 0.50  # Rs./kWh cycled (calendar + cycle aging proxy)

# ---------- Forecasting ----------
FORECAST_HISTORY_DAYS = 30      # days of synthetic history
FORECAST_DAY_VARIATION = 0.10   # +/- 10% day-to-day variation

# ---------- Chart Settings ----------
CHART_DPI = 300
CHART_DIR = "charts"

# Color scheme
COLOR_SOLAR = "#FFB300"         # amber/gold
COLOR_ORC = "#E65100"           # deep orange
COLOR_BATTERY = "#1565C0"       # blue
COLOR_GRID = "#757575"          # gray
COLOR_LOAD = "#212121"          # near-black
COLOR_DSM = "#2E7D32"           # green
COLOR_EXCESS = "#AB47BC"        # purple
COLOR_SOC = "#0288D1"           # light blue
