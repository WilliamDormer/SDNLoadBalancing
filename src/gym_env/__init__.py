from gymnasium.envs.registration import register

register(
    id="network_env/NetworkSim-v0",
    entry_point="gym_env.envs.network_sim_env:NetworkSimEnv",
    max_episode_steps=100,
)
