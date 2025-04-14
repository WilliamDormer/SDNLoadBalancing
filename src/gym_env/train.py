# this is the example of using a DQN provided by stable baselines3
# we need to modify this for our use with a custom environment for our network simulator

# TODO we also need to evaluate if this will fit with our slower, more complex simulator.
from gym_env.envs.network_sim_env import NetworkSimEnv
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import os
import argparse

# Set the default device to MPS (Metal Performance Shaders) for Apple Silicon
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")


# Define the DQN architecture with batch normalization for better performance
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # Add dropout for regularization
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use a smaller initialization scale
            nn.init.kaiming_normal_(module.weight, a=0.01, nonlinearity="relu")
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.network(x)


# Optimized ReplayBuffer with numpy arrays
class ReplayBuffer:
    def __init__(self, capacity, state_dim):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        # Pre-allocate memory for better performance
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

    def push(self, state, action, reward, next_state, done):
        # Check if buffer is full
        if self.size >= self.capacity:
            # Reset buffer
            self.position = 0
            self.size = 0
            print("ReplayBuffer full - clearing buffer")

            # Clear arrays
            self.states.fill(0)
            self.actions.fill(0)
            self.rewards.fill(0)
            self.next_states.fill(0)
            self.dones.fill(0)

        # Store transition
        self.states[self.position] = state.flatten()
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state.flatten()
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


# Adjusted training parameters for better stability and learning
BATCH_SIZE = 32  # Increased from 64 for more stable gradients
GAMMA = 0.99  # Keep as is
EPSILON_START = 1.0
EPSILON_END = 0.05  # Increased from 0.01 for more exploration
EPSILON_DECAY = 0.98  # Slower decay for better exploration
LEARNING_RATE = 1e-3  # Reduced from 1e-3 for more stable learning
MEMORY_SIZE = 10000  # Increased for better experience diversity
MIN_MEMORY_SIZE = 100  # Increased for better initial learning
TARGET_UPDATE = 5  # More frequent target updates
UPDATE_FREQ = 2  # Update every step

# Create the environment
env = NetworkSimEnv(
    render_mode="no",
    num_controllers=4,
    num_switches=26,
    max_rate=1000,
    gc_ip="localhost",
    gc_port="8000",
    step_time=0.5,
)

# Initialize networks
input_dim = env.m * env.n
output_dim = env.action_space.n

print(f"Input dimension: {input_dim}, Output dimension: {output_dim}")

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Initialize running_reward
running_reward = 0.0

# Use Adam optimizer with AMSGrad for better stability
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayBuffer(MEMORY_SIZE, input_dim)
epsilon = EPSILON_START

# Initialize statistics tracking
training_stats = {
    "episode_rewards": [],
    "episode_lengths": [],
    "average_losses": [],
    "running_rewards": [],
    "epsilons": [],
}


@torch.no_grad()  # Optimization decorator
def select_action(state, epsilon):
    if random.random() < epsilon:
        return env.action_space.sample()

    state = torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
    q_values = policy_net(state)
    return q_values.max(1)[1].item() + 1


def optimize_model():
    if len(memory) < MIN_MEMORY_SIZE:
        return

    states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

    # Normalize states for better training stability
    states = torch.FloatTensor(states).to(device)
    states = (states - states.mean(dim=0)) / (states.std(dim=0) + 1e-8)

    next_states = torch.FloatTensor(next_states).to(device)
    next_states = (next_states - next_states.mean(dim=0)) / (
        next_states.std(dim=0) + 1e-8
    )

    actions = torch.LongTensor(actions - 1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Double DQN implementation
    with torch.no_grad():
        next_actions = policy_net(next_states).max(1)[1]
        next_q_values = (
            target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
        )
        expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))

    # Use Huber loss with a smaller delta for better stability
    loss = nn.SmoothL1Loss(beta=0.5)(current_q_values.squeeze(), expected_q_values)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Stronger gradient clipping
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=0.5)
    optimizer.step()

    return loss.item()


# Add this function after the model initialization and before the training loop
def load_checkpoint(checkpoint_path):
    """
    Load a training checkpoint and return the starting episode number
    """
    if not os.path.exists(checkpoint_path):
        print(f"No checkpoint found at {checkpoint_path}")
        return 0

    # Load checkpoint with weights_only=False to handle all data types
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    policy_net.load_state_dict(checkpoint["model_state_dict"])
    target_net.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load training stats
    global epsilon, running_reward
    epsilon = float(checkpoint["epsilon"])  # Convert numpy float to Python float
    running_reward = float(
        checkpoint["running_reward"]
    )  # Convert numpy float to Python float

    print(f"Loaded checkpoint from episode {checkpoint['episode'] + 1}")
    print(f"Epsilon: {epsilon:.3f}, Running Reward: {running_reward:.2f}")

    return checkpoint["episode"] + 1


# Modify the training loop initialization to accept a starting episode
def train(start_episode=0, num_episodes=100):
    total_steps = 0
    global epsilon, running_reward  # Make these global so they persist between training sessions

    # Add progress bar for episodes
    with tqdm(
        total=start_episode + num_episodes,
        initial=start_episode,
        desc="Training Episodes",
    ) as pbar:
        for episode in range(start_episode, start_episode + num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_loss = 0
            episode_steps = 0

            # Skip training if state is all zeros (during warm-up)
            is_warmed_up = np.any(state != 0)

            # Add progress bar for steps within episode
            with tqdm(
                total=100, desc=f"Episode {episode+1} Steps", leave=False
            ) as step_pbar:
                for step in range(100):  # Max steps per episode
                    total_steps += 1
                    episode_steps += 1

                    # Only take trained actions if warmed up, otherwise random
                    if is_warmed_up:
                        action = select_action(state, epsilon)
                    else:
                        action = 0

                    next_state, reward, done, truncated, _ = env.step(action)

                    # Only store experiences and train if warmed up
                    if is_warmed_up:
                        memory.push(state, action, reward, next_state, done)
                        if total_steps % UPDATE_FREQ == 0:
                            loss = optimize_model()
                            if loss is not None:
                                episode_loss += loss
                                step_pbar.set_postfix(
                                    {
                                        "loss": f"{loss:.4f}",
                                        "buffer": f"{len(memory)}/{MIN_MEMORY_SIZE}",
                                        "reward": f"{reward:.2f}",
                                    }
                                )
                    else:
                        # Check if we've warmed up during this step
                        is_warmed_up = np.any(next_state != 0)
                        if is_warmed_up:
                            step_pbar.write(
                                "Flow statistics detected - starting training"
                            )

                    state = next_state
                    episode_reward += reward
                    step_pbar.update(1)

                    if done or truncated:
                        break

            # Update target network
            if episode % TARGET_UPDATE == 0:
                target_net.load_state_dict(policy_net.state_dict())

            # Decay epsilon
            epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

            # Update running reward
            running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

            # Save statistics at the end of each episode
            training_stats["episode_rewards"].append(episode_reward)
            training_stats["episode_lengths"].append(episode_steps)
            avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
            training_stats["average_losses"].append(avg_loss)
            training_stats["running_rewards"].append(running_reward)
            training_stats["epsilons"].append(epsilon)

            # Update episode progress bar with metrics
            pbar.set_postfix(
                {
                    "reward": f"{episode_reward:.2f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "epsilon": f"{epsilon:.3f}",
                }
            )
            pbar.update(1)

            # Save the model periodically
            if (episode + 1) % 10 == 0:
                torch.save(
                    {
                        "episode": episode,
                        "model_state_dict": policy_net.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "running_reward": running_reward,
                        "epsilon": epsilon,
                    },
                    f"dqn_model_episode_{episode + 1}.pth",
                )


# Save statistics to file
def save_stats(stats):
    import json

    # Try to load existing stats first
    try:
        with open("training_stats.json", "r") as f:
            existing_stats = json.load(f)
            # Combine existing stats with new stats
            for key in stats:
                stats[key] = existing_stats.get(key, []) + stats[key]
    except FileNotFoundError:
        pass  # No existing stats file, will create new one

    # Save combined stats
    with open("training_stats.json", "w") as f:
        json.dump(stats, f)

    return stats


def load_training_stats():
    """Load existing training statistics from file"""
    import json

    try:
        with open("training_stats.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


# After training loop, create and save plots
def plot_training_stats(stats, show_plot=False):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Add episode numbers to x-axis
    episodes = range(1, len(stats["episode_rewards"]) + 1)

    # Plot episode rewards
    ax1.plot(episodes, stats["episode_rewards"])
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")

    # Plot running rewards
    ax2.plot(episodes, stats["running_rewards"])
    ax2.set_title("Running Rewards")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Running Reward")

    # Plot average losses
    ax3.plot(episodes, stats["average_losses"])
    ax3.set_title("Average Losses")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Loss")

    # Plot epsilon decay
    ax4.plot(episodes, stats["epsilons"])
    ax4.set_title("Epsilon Decay")
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Epsilon")

    plt.tight_layout()
    plt.savefig("training_stats.png")
    if show_plot:
        plt.show()
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DQN for network load balancing")
    parser.add_argument(
        "--checkpoint", type=str, help="Path to checkpoint file to resume training"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to train"
    )
    parser.add_argument(
        "--show-plot", action="store_true", help="Show the plot after training"
    )
    args = parser.parse_args()

    # Initialize starting episode
    start_episode = 0

    # Load checkpoint if specified
    if args.checkpoint:
        start_episode = load_checkpoint(args.checkpoint)

    # Start training
    train(start_episode=start_episode, num_episodes=args.episodes)

    # Plot and save statistics
    stats = save_stats(training_stats)
    plot_training_stats(stats, show_plot=args.show_plot)

    env.close()
