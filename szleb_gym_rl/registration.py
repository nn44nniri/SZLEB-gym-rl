def register_szleb_env() -> None:
    """
    Register the environment as 'SZLEB-v0'.
    Safe to call multiple times.
    """
    try:
        import gymnasium as gym
    except ImportError:
        import gym  # type: ignore

    env_id = "SZLEB-v0"

    # Avoid duplicate registration errors
    try:
        registry = gym.envs.registry
        if env_id in registry:
            return
    except Exception:
        # Some gym versions expose registry differently; ignore and try register
        pass

    gym.register(
        id=env_id,
        entry_point="szleb_gym_rl.env:SZLEBGymEnv",
    )