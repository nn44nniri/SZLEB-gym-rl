
# szleb_gym_rl

A lightweight **Gym/Gymnasium reinforcement learning environment** built on top of the **SZLEB greenhouse estimator**.

This package wraps the SZLEB single-zone greenhouse energy + moisture simulator into an RL-friendly environment named:

* **`SZLEB-v0`**

It allows you to run a **cropping season table** (one row = one day) and obtain, for each row/day:

* final indoor temperature and humidity,
* airflow / ACH summaries,
* **separate energy consumption by actuator/operator**:

  * heater gas consumption (m³),
  * heater auxiliary electricity (kWh),
  * cooling fan electricity (kWh),
  * vent motor electricity (kWh),
  * total electricity (kWh).

---

## What this project does

The original SZLEB logic resolves actuator commands (ON/OFF, activity %, active time) into:

* **ACH** (ventilation / airflow proxy),
* **heater thermal power** (W),
* and **energy rates** for each actuator (gas/electric) before running the simulator. 

The simulator then performs a coupled time-step simulation (air temperature, canopy temperature, humidity/transpiration, ventilation heat exchange, solar and heater effects) and accumulates actuator energy consumption over time. 

`szleb_gym_rl` keeps that logic and simply exposes it as a **row-by-row RL environment**.

---

## Environment ID

* **`SZLEB-v0`**

---

## Main idea (RL wrapper behavior)

* **Input to env**: a pandas table for one cropping season (consecutive rows/days).
* **Each `step()`**:

  1. Reads one row (day, LAI, temperatures, actuator status),
  2. Calls SZLEB estimator for that row (typically one full day),
  3. Returns observation + reward + info,
  4. Stores a one-row summary (including separate actuator energy totals).

This makes it usable for:

* offline simulation over a fixed control schedule,
* RL training (if actions override row actuator columns),
* optimization benchmarking.

---

## Input table format (season table)

Each row represents **one day** of a consecutive cropping season.

### Required conceptual columns

* `day` (or `DOY`)
* `LAI` (leaf area index)
* `t_out_c` (outside air temperature, °C)
* `t_in_c` (inside target / initial air temperature for the day, °C)
* actuator status columns (see below)

### Actuator status columns (recommended)

* `vents_on` (0/1 or True/False)

* `vents_activity_pct` (0..100)

* `vents_active_time_s` (seconds active during the day)

* `fans_on`

* `fans_activity_pct`

* `fans_active_time_s`

* `heater_on`

* `heater_activity_pct`

* `heater_active_time_s`

### Optional columns (defaults can be used)

* `rh_out_pct`
* `rh_in_pct`
* `g_sun_w_m2`

---

## Output per row/day (what `step()` provides)

In `info["current_row_output"]`, the environment returns a structured summary for the current day, including:

* `Tin_final_C`
* `RHin_final_pct`
* `airflow_avg_m3_s`
* `airflow_final_m3_s`
* `ach_avg_1_h`
* `ach_final_1_h`

### Separate energy consumption by operator (requested output)

* `heater_gas_total_m3`
* `heater_elec_total_kwh`
* `cooling_elec_total_kwh`  *(cooling actuator = fans in this abstraction)*
* `vents_elec_total_kwh`
* `total_elec_kwh`

The full season summary can be exported as a dataframe using:

* `env.unwrapped.get_history_dataframe()`

---

## Action space (optional RL control mode)

The environment supports two modes:

### 1) **Table-driven mode** (default)

Actuator statuses are read directly from each row of the season table.

* `use_action_override=False`

### 2) **Agent-driven mode** (RL control)

The action overrides row actuator statuses.

* `use_action_override=True`

Action vector (continuous, normalized `[0,1]`) length = **9**:

* vents: on, activity, duty
* fans: on, activity, duty
* heater: on, activity, duty

Where:

* `on > 0.5` means ON
* `activity ∈ [0,1]` → mapped to `0..100%`
* `duty ∈ [0,1]` → mapped to active time during the row elapsed period

---

## Observation space (summary)

A compact observation vector is returned, including (conceptually):

* normalized day,
* LAI,
* outside temperature,
* inside target temperature,
* previous step final indoor temperature,
* previous step final humidity,
* previous step total electricity,
* previous step heater gas.

This is intentionally simple and can be extended later.

---

## Quick start (example)

```python
import pandas as pd
import gymnasium as gym  # or gym
from szleb_gym_rl import register_szleb_env, SZLEBEnvConfig

register_szleb_env()

season_df = pd.DataFrame([
    {
        "day": 100, "LAI": 1.5,
        "t_out_c": -5.0, "t_in_c": 18.0,
        "rh_out_pct": 70.0, "rh_in_pct": 60.0, "g_sun_w_m2": 120.0,
        "vents_on": 0, "vents_activity_pct": 0.0, "vents_active_time_s": 0.0,
        "fans_on": 1,  "fans_activity_pct": 30.0, "fans_active_time_s": 12*3600,
        "heater_on": 1,"heater_activity_pct": 70.0, "heater_active_time_s": 10*3600,
    }
])

cfg = SZLEBEnvConfig(use_action_override=False)

env = gym.make("SZLEB-v0", season_table=season_df, config=cfg)

obs, info = env.reset()
action = env.action_space.sample()  # ignored in table-driven mode
obs, reward, terminated, truncated, info = env.step(action)

print(info["current_row_output"])   # per-day separate energy outputs
print(env.unwrapped.get_history_dataframe())
```

---

## Project structure (suggested)

```text
szleb/
├── greenhouse_estimator/
│   ├── actuators.py
│   ├── models.py
│   ├── psychrometrics.py
│   ├── simulator.py
│   └── __init__.py
├── main_example.py
└── README.md

szleb_gym_rl/
├── __init__.py
├── registration.py
├── config.py
├── utils.py
└── env.py

main_szleb_v0.py
```

---

## Notes on SZLEB physics and bookkeeping

This RL wrapper does **not replace** SZLEB physics; it reuses it.

The base SZLEB code already includes:

* actuator abstraction (`ActuatorCommand`, `ActuatorSet`, `ActuatorMapping`),
* actuator-to-ACH/heater-power mapping,
* psychrometric utilities,
* coupled air/canopy temperature + humidity simulation,
* per-step and cumulative actuator energy bookkeeping. 

The wrapper simply organizes those calculations into a Gym-compatible episode over a seasonal table.

---

## Typical use cases

* **Replay fixed greenhouse control schedules** over a season table
* **Energy analysis** (per operator/per day)
* **RL experiments** for actuator control
* **Reward design** for temperature tracking vs. energy cost trade-offs
* **Benchmarking** control policies before using more detailed simulators

---

## Limitations (current version)

* Single-zone abstraction (not CFD / spatial velocity field)
* “Wind speed” can only be approximated from airflow and assumed opening area (proxy)
* Reward function is intentionally simple by default
* Row granularity is usually one day (can be adapted to hourly rows)

---

## Next extensions (recommended)

* Hourly-row environment (`1 row = 1 hour`) for finer control
* Multi-objective rewards (temperature, RH, cost, gas/electric split)
* Constraint penalties (comfort/agronomic ranges)
* Domain randomization for robust RL training
* Vectorized environments for faster training

---

