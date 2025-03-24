# this is the example of using a DQN provided by stable baselines3
# we need to modify this for our use with a custom environment for our network simulator

# TODO we also need to evaluate if this will fit with our slower, more complex simulator.
from gym_env.envs.network_sim_env import NetworkSimEnv
# Create the environment
env = NetworkSimEnv(
    render_mode="human",
    num_controllers=4,
    num_switches=26,
    max_rate=5000,
    gc_ip="localhost",
    gc_port="8000",
    step_time=5,
)

# Run a few test episodes
n_episodes = 3
max_steps = 100

for episode in range(n_episodes):
    obs, info = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        # Take random actions for testing
        action = (0,0)

        # Perform step in environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward

        print(f"Episode {episode + 1}, Step {step + 1}")
        print(f"Action taken: {action}")
        print(f"Reward: {reward}")
        print(f"Current observation: {obs}")
        print("-" * 50)

        if terminated or truncated:
            print(f"Episode {episode + 1} finished after {step + 1} steps")
            print(f"Total reward: {episode_reward}")
            print("=" * 50)
            break

env.close()
