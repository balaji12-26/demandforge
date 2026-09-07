"""
DemandForge Models
==================
Physical models for solar PV, ORC waste-to-energy, load profiles, and BESS.
"""

import numpy as np
import config as cfg


def solar_generation(hour: float, capacity_kw: float = cfg.SOLAR_CAPACITY_KWP) -> float:
    """
    Solar PV output using a Gaussian bell curve centred at solar noon.
    Returns 0 outside sunrise-sunset window.
    """
    if hour < cfg.SOLAR_SUNRISE_HOUR or hour > cfg.SOLAR_SUNSET_HOUR:
        return 0.0
    # Gaussian with sigma chosen so output is ~5% of peak at sunrise/sunset
    sigma = (cfg.SOLAR_SUNSET_HOUR - cfg.SOLAR_SUNRISE_HOUR) / 4.5
    peak_output = capacity_kw * cfg.SOLAR_EFFICIENCY
    output = peak_output * np.exp(-0.5 * ((hour - cfg.SOLAR_PEAK_HOUR) / sigma) ** 2)
    return max(0.0, output)


def orc_generation(hour: float, capacity_kw: float = cfg.ORC_CAPACITY_KW) -> float:
    """
    ORC waste-to-energy output.
    Constant during factory operating hours, reduced at night.
    """
    if cfg.ORC_FACTORY_START <= hour < cfg.ORC_FACTORY_END:
        return capacity_kw
    else:
        return capacity_kw * cfg.ORC_NIGHT_FRACTION


def get_tariff(hour: float) -> float:
    """Return the effective tariff (Rs./kWh) at a given hour, including ToD surcharge."""
    base = cfg.BASE_TARIFF_PER_KWH
    if cfg.PEAK_START_HOUR <= hour < cfg.PEAK_END_HOUR:
        return base * (1.0 + cfg.TOD_SURCHARGE)
    return base


def generate_load_profile(
    base_kw: float = cfg.BASE_LOAD_KW,
    peak_factor: float = cfg.PEAK_FACTOR,
    with_demand_side: bool = False,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a realistic 96-element industrial load profile (kW) for 24 hours.

    Segments:
        0-6 AM   : 400 kW (night shift)
        6-8 AM   : ramp 400 -> 1000 kW
        8 AM-5 PM: 1000 kW baseline
        5-6 PM   : ramp 1000 -> peak
        6-10 PM  : peak with transients
        10 PM-12 : ramp down to 600 kW

    With demand-side management the evening peak is flattened and some load
    is shifted to solar hours.
    """
    rng = np.random.RandomState(seed)
    load = np.zeros(cfg.TOTAL_STEPS)
    peak_kw = base_kw * peak_factor

    for i in range(cfg.TOTAL_STEPS):
        hour = i * cfg.DT_HOURS

        if hour < 6.0:
            # Night shift
            load[i] = cfg.NIGHT_LOAD_KW
        elif hour < 8.0:
            # Morning ramp-up
            frac = (hour - 6.0) / 2.0
            load[i] = cfg.NIGHT_LOAD_KW + frac * (base_kw - cfg.NIGHT_LOAD_KW)
        elif hour < 17.0:
            # Daytime baseline
            load[i] = base_kw
        elif hour < 18.0:
            # Evening ramp to peak
            frac = (hour - 17.0) / 1.0
            load[i] = base_kw + frac * (peak_kw - base_kw)
        elif hour < 22.0:
            # Peak period with spiky transients (motor inrush etc.)
            spike = rng.uniform(0.0, 0.08) * peak_kw  # up to 8% spikes
            load[i] = peak_kw + spike
        else:
            # Post-peak ramp-down
            frac = (hour - 22.0) / 2.0
            load[i] = peak_kw - frac * (peak_kw - cfg.EVENING_RAMPDOWN_KW)

        # Small noise on all steps
        load[i] += rng.normal(0, 8)

    # Ensure positive
    load = np.maximum(load, 50.0)

    if with_demand_side:
        load = _apply_demand_side(load, rng)

    return load


def _apply_demand_side(load: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """
    Apply demand-side management strategies:
    1. Motor-start staggering: reduce peak-period spikes
    2. Thermal pre-loading: shift evening kWh to solar hours
    3. Pump/compressor scheduling: shift to solar hours
    """
    modified = load.copy()

    # Identify peak steps (6-10 PM = steps 72..87)
    peak_start_step = int(cfg.PEAK_START_HOUR * cfg.STEPS_PER_HOUR)
    peak_end_step = int(cfg.PEAK_END_HOUR * cfg.STEPS_PER_HOUR)

    # Also pre-peak ramp (5-6 PM = steps 68..71)
    ramp_start_step = int(17.0 * cfg.STEPS_PER_HOUR)

    # Solar hours for load shifting (8 AM - 4 PM = steps 32..63)
    solar_start_step = int(8.0 * cfg.STEPS_PER_HOUR)
    solar_end_step = int(16.0 * cfg.STEPS_PER_HOUR)
    n_solar_steps = solar_end_step - solar_start_step

    # 1. Motor staggering: reduce peak steps
    for i in range(peak_start_step, peak_end_step):
        reduction = modified[i] * cfg.DSM_PEAK_REDUCTION
        modified[i] -= reduction

    # 2. Thermal pre-loading: shift evening energy to solar hours
    total_shifted = 0.0
    for i in range(ramp_start_step, peak_end_step):
        shift = modified[i] * cfg.DSM_LOAD_SHIFT_FRACTION
        modified[i] -= shift
        total_shifted += shift

    # Distribute shifted load across solar hours
    shift_per_step = total_shifted / n_solar_steps
    for i in range(solar_start_step, solar_end_step):
        modified[i] += shift_per_step

    # 3. Scheduling: small additional shift
    for i in range(peak_start_step, peak_end_step):
        sched_shift = modified[i] * cfg.DSM_SCHEDULING_REDUCTION
        modified[i] -= sched_shift
        # add to solar hours
        idx = rng.randint(solar_start_step, solar_end_step)
        modified[idx] += sched_shift

    return modified


class BatteryModel:
    """
    Lithium Iron Phosphate (LFP) battery energy storage system.
    Tracks SOC and enforces charge/discharge limits.
    """

    def __init__(
        self,
        capacity_kwh: float = cfg.BESS_NOMINAL_KWH,
        soc_min: float = cfg.SOC_MIN,
        soc_max: float = cfg.SOC_MAX,
        soc_initial: float = cfg.SOC_INITIAL,
        charge_eff: float = cfg.CHARGE_EFFICIENCY,
        discharge_eff: float = cfg.DISCHARGE_EFFICIENCY,
        max_charge_kw: float = cfg.MAX_CHARGE_RATE_KW,
        max_discharge_kw: float = cfg.MAX_DISCHARGE_RATE_KW,
    ):
        self.capacity_kwh = capacity_kwh
        self.soc_min = soc_min
        self.soc_max = soc_max
        self.soc = soc_initial
        self.charge_eff = charge_eff
        self.discharge_eff = discharge_eff
        self.max_charge_kw = max_charge_kw
        self.max_discharge_kw = max_discharge_kw

    @property
    def energy_kwh(self) -> float:
        return self.soc * self.capacity_kwh

    @property
    def available_discharge_kwh(self) -> float:
        """Energy available to discharge (accounting for SOC min)."""
        return (self.soc - self.soc_min) * self.capacity_kwh

    @property
    def available_charge_kwh(self) -> float:
        """Room to charge (accounting for SOC max)."""
        return (self.soc_max - self.soc) * self.capacity_kwh

    def charge(self, power_kw: float, dt_hours: float = cfg.DT_HOURS) -> float:
        """
        Charge the battery. Returns actual power accepted (kW).
        Accounts for charging efficiency and SOC ceiling.
        """
        power_kw = min(power_kw, self.max_charge_kw)
        power_kw = max(power_kw, 0.0)

        # Energy that would be stored (after efficiency loss)
        energy_in = power_kw * dt_hours * self.charge_eff
        room = self.available_charge_kwh
        if energy_in > room:
            energy_in = room
            power_kw = energy_in / (dt_hours * self.charge_eff)

        self.soc += energy_in / self.capacity_kwh
        self.soc = min(self.soc, self.soc_max)
        return power_kw

    def discharge(self, power_kw: float, dt_hours: float = cfg.DT_HOURS) -> float:
        """
        Discharge the battery. Returns actual power delivered (kW).
        Accounts for discharge efficiency and SOC floor.
        """
        power_kw = min(power_kw, self.max_discharge_kw)
        power_kw = max(power_kw, 0.0)

        # Energy drawn from battery (before efficiency loss)
        energy_drawn = power_kw * dt_hours / self.discharge_eff
        available = self.available_discharge_kwh
        if energy_drawn > available:
            energy_drawn = available
            power_kw = energy_drawn * self.discharge_eff / dt_hours

        self.soc -= energy_drawn / self.capacity_kwh
        self.soc = max(self.soc, self.soc_min)
        return power_kw

    def reset(self, soc: float = None):
        """Reset battery to given SOC (or initial)."""
        self.soc = soc if soc is not None else cfg.SOC_INITIAL


def apply_demand_side_management(load: np.ndarray, seed: int = 42) -> np.ndarray:
    """Public wrapper for demand-side management strategies."""
    rng = np.random.RandomState(seed)
    return _apply_demand_side(load, rng)

