from dataclasses import dataclass, field
from typing import Optional

# Import SZLEB types from original library
from szleb.greenhouse_estimator import (
    Geometry, BuildingParams, CouplingParams, ActuatorMapping
)


@dataclass
class SZLEBEnvConfig:
    """
    Default configuration for the SZLEB Gym environment.
    Rows can override several values via table columns (see utils.py).
    """
    # Simulation timing
    dt_s: float = 60.0                  # simulator internal step (seconds)
    elapsed_s_per_row: float = 24 * 3600.0  # one row = one full day by default

    # Defaults for missing columns
    default_rh_out_pct: float = 70.0
    default_rh_in_pct: float = 60.0
    default_g_sun_w_m2: float = 150.0
    default_crop_area_m2: Optional[float] = None  # None => floor area

    # Geometry / building / coupling baseline
    geometry: Geometry = field(default_factory=lambda: Geometry(L=20.0, W=10.0, H=4.0))
    building: BuildingParams = field(default_factory=lambda: BuildingParams(
        UA_w_k=250.0,
        tau_alpha_air=0.25,
        tau_alpha_can=0.20,
    ))
    actuator_mapping: ActuatorMapping = field(default_factory=lambda: ActuatorMapping(
        base_ach_vents=3.0,
        base_ach_fans=6.0,
        heater_max_w=20_000.0,
        heater_max_gas_m3_h=2.8,
        heater_aux_elec_max_w=250.0,
        cooling_fans_max_elec_w=1800.0,
        vents_motor_max_elec_w=80.0,
    ))

    # RL reward weights (simple default)
    w_temp_tracking: float = 1.0
    w_total_elec_kwh: float = 0.2
    w_heater_gas_m3: float = 0.2

    # Action behavior
    # If True, env.step(action) overrides actuator fields from row.
    # If False, action can be None and row actuator columns are used directly.
    use_action_override: bool = False