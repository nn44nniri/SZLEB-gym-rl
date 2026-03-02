from __future__ import annotations
from typing import Any, Dict

import math


def _to_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    s = str(x).strip().lower()
    if s in {"1", "true", "on", "yes", "y"}:
        return True
    if s in {"0", "false", "off", "no", "n"}:
        return False
    return default


def _to_float(x: Any, default: float) -> float:
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def normalize_day_value(day_val: float) -> float:
    # robust 0..1 normalization for "day-like" values
    if math.isnan(day_val):
        return 0.0
    return clamp(day_val / 366.0, 0.0, 1.0)


def parse_row_inputs(
    row: Dict[str, Any],
    elapsed_s_per_row: float,
    defaults: Dict[str, float],
) -> Dict[str, Any]:
    """
    Parse one table row into SZLEB-ready inputs.
    Required conceptual fields:
      - day / doy (any one of aliases)
      - lai
      - t_out_c
      - t_in_c (used as initial indoor target / starting state)

    Actuator statuses can come from row (if action override is disabled).

    Supported aliases (examples):
      day: day, doy, Day, DOY
      LAI: lai, LAI
      outside temp: t_out_c, Tout_C, outside_temp_c
      inside temp: t_in_c, Tin_C, inside_temp_c
      RH outside: rh_out_pct, RH_out_pct
      RH inside: rh_in_pct, RH_in_pct
      G_sun: g_sun_w_m2, G_sun_w_m2, solar_w_m2
    """
    # --- Core scalar inputs ---
    day = _to_float(
        row.get("day", row.get("doy", row.get("Day", row.get("DOY", 1)))),
        1.0
    )
    lai = _to_float(row.get("lai", row.get("LAI", 1.0)), 1.0)

    t_out_c = _to_float(
        row.get("t_out_c", row.get("Tout_C", row.get("outside_temp_c", 0.0))),
        0.0
    )
    t_in_c = _to_float(
        row.get("t_in_c", row.get("Tin_C", row.get("inside_temp_c", 20.0))),
        20.0
    )

    rh_out_pct = _to_float(
        row.get("rh_out_pct", row.get("RH_out_pct", defaults["rh_out_pct"])),
        defaults["rh_out_pct"]
    )
    rh_in_pct = _to_float(
        row.get("rh_in_pct", row.get("RH_in_pct", defaults["rh_in_pct"])),
        defaults["rh_in_pct"]
    )
    g_sun_w_m2 = _to_float(
        row.get("g_sun_w_m2", row.get("G_sun_w_m2", row.get("solar_w_m2", defaults["g_sun_w_m2"]))),
        defaults["g_sun_w_m2"]
    )

    # --- Actuators from row (used when no action override) ---
    # on/off
    vents_on = _to_bool(row.get("vents_on", row.get("vent_on", False)), False)
    fans_on = _to_bool(row.get("fans_on", row.get("fan_on", False)), False)
    heater_on = _to_bool(row.get("heater_on", False), False)

    # activity 0..100
    vents_activity_pct = clamp(_to_float(row.get("vents_activity_pct", row.get("vent_activity_pct", 0.0)), 0.0), 0.0, 100.0)
    fans_activity_pct = clamp(_to_float(row.get("fans_activity_pct", row.get("fan_activity_pct", 0.0)), 0.0), 0.0, 100.0)
    heater_activity_pct = clamp(_to_float(row.get("heater_activity_pct", 0.0), 0.0), 0.0, 100.0)

    # active times (seconds in current row/day)
    vents_active_time_s = clamp(_to_float(row.get("vents_active_time_s", elapsed_s_per_row), elapsed_s_per_row), 0.0, elapsed_s_per_row)
    fans_active_time_s = clamp(_to_float(row.get("fans_active_time_s", elapsed_s_per_row), elapsed_s_per_row), 0.0, elapsed_s_per_row)
    heater_active_time_s = clamp(_to_float(row.get("heater_active_time_s", elapsed_s_per_row), elapsed_s_per_row), 0.0, elapsed_s_per_row)

    return {
        "day": day,
        "lai": lai,
        "t_out_c": t_out_c,
        "t_in_c": t_in_c,
        "rh_out_pct": rh_out_pct,
        "rh_in_pct": rh_in_pct,
        "g_sun_w_m2": g_sun_w_m2,
        "vents_on": vents_on,
        "vents_activity_pct": vents_activity_pct,
        "vents_active_time_s": vents_active_time_s,
        "fans_on": fans_on,
        "fans_activity_pct": fans_activity_pct,
        "fans_active_time_s": fans_active_time_s,
        "heater_on": heater_on,
        "heater_activity_pct": heater_activity_pct,
        "heater_active_time_s": heater_active_time_s,
    }


def action_to_actuators(action, elapsed_s_per_row: float) -> Dict[str, Any]:
    """
    Convert a continuous action vector into actuator commands.

    Action format (length 9):
      [0] vents_on01
      [1] vents_activity01
      [2] vents_duty01
      [3] fans_on01
      [4] fans_activity01
      [5] fans_duty01
      [6] heater_on01
      [7] heater_activity01
      [8] heater_duty01

    on01 > 0.5 => ON
    activity01 in [0,1] => 0..100 %
    duty01 in [0,1] => active_time_s = duty01 * elapsed_s_per_row
    """
    if action is None:
        raise ValueError("action is None but use_action_override=True")

    a = [float(x) for x in action]
    if len(a) != 9:
        raise ValueError(f"Expected action length 9, got {len(a)}")

    a = [clamp(x, 0.0, 1.0) for x in a]

    return {
        "vents_on": a[0] > 0.5,
        "vents_activity_pct": 100.0 * a[1],
        "vents_active_time_s": elapsed_s_per_row * a[2],
        "fans_on": a[3] > 0.5,
        "fans_activity_pct": 100.0 * a[4],
        "fans_active_time_s": elapsed_s_per_row * a[5],
        "heater_on": a[6] > 0.5,
        "heater_activity_pct": 100.0 * a[7],
        "heater_active_time_s": elapsed_s_per_row * a[8],
    }