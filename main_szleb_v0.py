import numpy as np
import pandas as pd

# Gym / Gymnasium compatibility
try:
    import gymnasium as gym
    GYMNASIUM = True
except ImportError:
    import gym  # type: ignore
    GYMNASIUM = False

from szleb_gym_rl import register_szleb_env, SZLEBEnvConfig


def build_example_season_table(n_days: int = 7) -> pd.DataFrame:
    """
    Example table:
    each row = one day of consecutive cropping season
    Includes:
      - day (DOY-like)
      - LAI
      - outside / inside temps
      - actuator statuses
    """
    rows = []
    for i in range(n_days):
        day = 100 + i
        rows.append({
            "day": day,
            "LAI": 1.2 + 0.1 * i,
            "t_out_c": -5.0 + 0.8 * i,    # outside temperature
            "t_in_c": 18.0 + 0.2 * i,     # desired / initial inside temperature

            # Optional humidity/solar columns (if omitted, env config defaults are used)
            "rh_out_pct": 65.0,
            "rh_in_pct": 60.0,
            "g_sun_w_m2": 120.0 + 10.0 * i,

            # Actuator status columns (used when use_action_override=False)
            "vents_on": 1 if i % 3 == 0 else 0,
            "vents_activity_pct": 20.0 if i % 3 == 0 else 0.0,
            "vents_active_time_s": 8 * 3600.0 if i % 3 == 0 else 0.0,

            "fans_on": 1,
            "fans_activity_pct": 30.0 + i * 2.0,
            "fans_active_time_s": 12 * 3600.0,

            "heater_on": 1 if i < 5 else 0,
            "heater_activity_pct": 70.0 if i < 5 else 0.0,
            "heater_active_time_s": 10 * 3600.0 if i < 5 else 0.0,
        })
    return pd.DataFrame(rows)


def main():
    register_szleb_env()

    season_df = build_example_season_table(n_days=10)

    # Config: use actuator values from table (no RL action override)
    cfg = SZLEBEnvConfig(
        elapsed_s_per_row=24 * 3600.0,   # one full day per row
        dt_s=60.0,
        use_action_override=False,       # IMPORTANT: row actuator status drives simulation
    )

    env = gym.make("SZLEB-v0", season_table=season_df, config=cfg)

    # reset
    if GYMNASIUM:
        obs, info = env.reset()
        print("reset info:", info)
    else:
        obs = env.reset()

    print("initial obs:", obs)

    done = False
    step_idx = 0

    while not done:
        # Since use_action_override=False, action can be anything or None.
        # To stay compatible with wrappers, we pass a valid action sample.
        action = env.action_space.sample()

        if GYMNASIUM:
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        else:
            obs, reward, done, info = env.step(action)

        row_out = info["current_row_output"]
        print(f"\nDay step {step_idx} output:")
        print({
            "day": row_out["day"],
            "LAI": row_out["LAI"],
            "Tin_final_C": row_out["Tin_final_C"],
            "heater_gas_total_m3": row_out["heater_gas_total_m3"],
            "heater_elec_total_kwh": row_out["heater_elec_total_kwh"],
            "cooling_elec_total_kwh": row_out["cooling_elec_total_kwh"],
            "vents_elec_total_kwh": row_out["vents_elec_total_kwh"],
            "total_elec_kwh": row_out["total_elec_kwh"],
            "reward": reward,
        })

        step_idx += 1

    # Access the accumulated per-row/day summary table
    # With gym wrappers, use env.unwrapped
    history_df = env.unwrapped.get_history_dataframe()

    print("\n=== Season per-row/day energy summary ===")
    print(history_df)

    # Example: save to CSV
    history_df.to_csv("szleb_v0_energy_by_day.csv", index=False)
    print("\nSaved: szleb_v0_energy_by_day.csv")


if __name__ == "__main__":
    main()