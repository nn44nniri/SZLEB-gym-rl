from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Gym / Gymnasium compatibility
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYMNASIUM = True
except ImportError:
    import gym  # type: ignore
    from gym import spaces  # type: ignore
    GYMNASIUM = False

from szleb.greenhouse_estimator import (
    Geometry, Outside, Initial, BuildingParams, CouplingParams,
    estimate_environment
)

from .config import SZLEBEnvConfig
from .utils import parse_row_inputs, action_to_actuators, normalize_day_value


class SZLEBGymEnv(gym.Env):
    """
    Gym-style environment that wraps the SZLEB estimator row-by-row.

    Input at construction/reset:
      - season_table: pandas DataFrame (one row = one day)

    Observation (float vector):
      [day_norm, LAI, T_out_C, T_in_target_C, last_T_in_final_C, last_RH_final_pct, last_total_elec_kwh, last_heater_gas_m3]

    Action (optional, continuous Box[0,1]^9):
      If config.use_action_override=True, action overrides row actuator status.
      Otherwise, row actuator status is used and action may be None.

    Output in info:
      - per-day separated energy consumption (heater gas/elec, cooling elec, vents elec)
      - full SZLEB result (including time-series rows)
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        season_table: Optional[pd.DataFrame] = None,
        config: Optional[SZLEBEnvConfig] = None,
    ):
        super().__init__()
        self.config = config or SZLEBEnvConfig()
        self.season_table: pd.DataFrame = pd.DataFrame() if season_table is None else season_table.copy()

        # Continuous action space (0..1), can be ignored if use_action_override=False
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(9,), dtype=np.float32
        )

        # Observation vector
        # [day_norm, LAI, T_out, T_in_target, last_Tin_final, last_RH_final, last_total_elec, last_heater_gas]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -50.0, -50.0, -50.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 15.0, 60.0, 60.0, 60.0, 100.0, 1e6, 1e6], dtype=np.float32),
            dtype=np.float32
        )

        self._idx: int = 0
        self._history: List[Dict[str, Any]] = []
        self._last_result: Optional[Dict[str, Any]] = None

    # -------------------------
    # Gym API
    # -------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        if GYMNASIUM:
            super().reset(seed=seed)
        else:
            # gym legacy
            if seed is not None and hasattr(self, "seed"):
                self.seed(seed)

        if options is not None and "season_table" in options and options["season_table"] is not None:
            self.season_table = options["season_table"].copy()

        if self.season_table is None or len(self.season_table) == 0:
            raise ValueError("season_table is empty. Pass a DataFrame at init or reset(options={'season_table': df}).")

        self._idx = 0
        self._history = []
        self._last_result = None

        obs = self._build_observation_for_current_row()
        info = {
            "current_row_index": self._idx,
            "rows_total": len(self.season_table),
        }

        if GYMNASIUM:
            return obs, info
        return obs

    def step(self, action):
        if self._idx >= len(self.season_table):
            raise RuntimeError("Episode is done. Call reset().")

        row_dict = self.season_table.iloc[self._idx].to_dict()

        parsed = parse_row_inputs(
            row=row_dict,
            elapsed_s_per_row=self.config.elapsed_s_per_row,
            defaults={
                "rh_out_pct": self.config.default_rh_out_pct,
                "rh_in_pct": self.config.default_rh_in_pct,
                "g_sun_w_m2": self.config.default_g_sun_w_m2,
            },
        )

        # Actuators: from action or row
        if self.config.use_action_override:
            actuator_fields = action_to_actuators(action, self.config.elapsed_s_per_row)
        else:
            actuator_fields = {
                "vents_on": parsed["vents_on"],
                "vents_activity_pct": parsed["vents_activity_pct"],
                "vents_active_time_s": parsed["vents_active_time_s"],
                "fans_on": parsed["fans_on"],
                "fans_activity_pct": parsed["fans_activity_pct"],
                "fans_active_time_s": parsed["fans_active_time_s"],
                "heater_on": parsed["heater_on"],
                "heater_activity_pct": parsed["heater_activity_pct"],
                "heater_active_time_s": parsed["heater_active_time_s"],
            }

        # Build SZLEB inputs
        geom: Geometry = self.config.geometry
        building: BuildingParams = self.config.building
        coupling = CouplingParams(LAI=float(parsed["lai"]))

        outside = Outside(
            T_out_c=float(parsed["t_out_c"]),
            RH_out_pct=float(parsed["rh_out_pct"]),
            G_sun_w_m2=float(parsed["g_sun_w_m2"]),
        )

        init = Initial(
            T_air_c=float(parsed["t_in_c"]),
            RH_air_pct=float(parsed["rh_in_pct"]),
            T_canopy_c=float(parsed["t_in_c"]),  # simple default coupling
        )

        result = estimate_environment(
            geom=geom,
            outside=outside,
            init=init,
            building=building,
            coupling=coupling,
            mapping=self.config.actuator_mapping,
            vents_on=bool(actuator_fields["vents_on"]),
            vents_activity_pct=float(actuator_fields["vents_activity_pct"]),
            vents_active_time_s=float(actuator_fields["vents_active_time_s"]),
            fans_on=bool(actuator_fields["fans_on"]),
            fans_activity_pct=float(actuator_fields["fans_activity_pct"]),
            fans_active_time_s=float(actuator_fields["fans_active_time_s"]),
            heater_on=bool(actuator_fields["heater_on"]),
            heater_activity_pct=float(actuator_fields["heater_activity_pct"]),
            heater_active_time_s=float(actuator_fields["heater_active_time_s"]),
            elapsed_s=float(self.config.elapsed_s_per_row),
            dt_s=float(self.config.dt_s),
            crop_area_m2=self.config.default_crop_area_m2,
        )

        self._last_result = result

        # Per-row/day separated outputs (the main requested output)
        row_output = {
            "row_index": self._idx,
            "day": parsed["day"],
            "LAI": parsed["lai"],
            "Tout_C": parsed["t_out_c"],
            "Tin_target_C": parsed["t_in_c"],
            "Tin_final_C": result["Tin_final_C"],
            "RHin_final_pct": result["RHin_final_pct"],

            # Separate energy by operator/actuator
            "heater_gas_total_m3": result["heater_gas_total_m3"],
            "heater_elec_total_kwh": result["heater_elec_total_kwh"],
            "cooling_elec_total_kwh": result["cooling_elec_total_kwh"],  # fans
            "vents_elec_total_kwh": result["vents_elec_total_kwh"],
            "total_elec_kwh": result["total_elec_kwh"],

            # Optional airflow summaries
            "airflow_avg_m3_s": result["airflow_avg_m3_s"],
            "airflow_final_m3_s": result["airflow_final_m3_s"],
            "ach_avg_1_h": result["ach_avg_1_h"],
            "ach_final_1_h": result["ach_final_1_h"],
        }
        self._history.append(row_output)

        # Reward (simple default): minimize energy + temperature tracking error
        temp_err = abs(float(result["Tin_final_C"]) - float(parsed["t_in_c"]))
        reward = -(
            self.config.w_temp_tracking * temp_err
            + self.config.w_total_elec_kwh * float(result["total_elec_kwh"])
            + self.config.w_heater_gas_m3 * float(result["heater_gas_total_m3"])
        )

        # Advance pointer
        self._idx += 1
        terminated = self._idx >= len(self.season_table)
        truncated = False

        if not terminated:
            obs = self._build_observation_for_current_row()
        else:
            # terminal observation: reuse last observation shape
            obs = self._terminal_observation()

        info = {
            "current_row_output": row_output,     # requested per-row separated energy
            "history_len": len(self._history),
            "szleb_result": result,               # includes internal time-series rows
        }

        if GYMNASIUM:
            return obs, float(reward), terminated, truncated, info
        else:
            done = terminated
            return obs, float(reward), done, info

    # -------------------------
    # Helpers
    # -------------------------
    def get_history_dataframe(self) -> pd.DataFrame:
        """Return one row/day summary outputs accumulated so far."""
        if len(self._history) == 0:
            return pd.DataFrame()
        return pd.DataFrame(self._history)

    def _build_observation_for_current_row(self) -> np.ndarray:
        row = self.season_table.iloc[self._idx].to_dict()
        parsed = parse_row_inputs(
            row=row,
            elapsed_s_per_row=self.config.elapsed_s_per_row,
            defaults={
                "rh_out_pct": self.config.default_rh_out_pct,
                "rh_in_pct": self.config.default_rh_in_pct,
                "g_sun_w_m2": self.config.default_g_sun_w_m2,
            },
        )

        if self._last_result is None:
            last_tin = float(parsed["t_in_c"])
            last_rh = float(parsed["rh_in_pct"])
            last_total_elec = 0.0
            last_heater_gas = 0.0
        else:
            last_tin = float(self._last_result["Tin_final_C"])
            last_rh = float(self._last_result["RHin_final_pct"])
            last_total_elec = float(self._last_result["total_elec_kwh"])
            last_heater_gas = float(self._last_result["heater_gas_total_m3"])

        obs = np.array([
            normalize_day_value(float(parsed["day"])),
            float(parsed["lai"]),
            float(parsed["t_out_c"]),
            float(parsed["t_in_c"]),
            last_tin,
            last_rh,
            last_total_elec,
            last_heater_gas,
        ], dtype=np.float32)
        return obs

    def _terminal_observation(self) -> np.ndarray:
        if self._last_result is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        # keep shape consistent
        obs = np.array([
            1.0,
            0.0,
            0.0,
            0.0,
            float(self._last_result["Tin_final_C"]),
            float(self._last_result["RHin_final_pct"]),
            float(self._last_result["total_elec_kwh"]),
            float(self._last_result["heater_gas_total_m3"]),
        ], dtype=np.float32)
        return obs

    def render(self, mode="human"):
        if mode != "human":
            return None
        if self._last_result is None:
            print("SZLEB-v0: no step has been run yet.")
            return
        print("SZLEB-v0 last result:")
        print({
            "Tin_final_C": self._last_result.get("Tin_final_C"),
            "RHin_final_pct": self._last_result.get("RHin_final_pct"),
            "heater_gas_total_m3": self._last_result.get("heater_gas_total_m3"),
            "heater_elec_total_kwh": self._last_result.get("heater_elec_total_kwh"),
            "cooling_elec_total_kwh": self._last_result.get("cooling_elec_total_kwh"),
            "vents_elec_total_kwh": self._last_result.get("vents_elec_total_kwh"),
            "total_elec_kwh": self._last_result.get("total_elec_kwh"),
        })