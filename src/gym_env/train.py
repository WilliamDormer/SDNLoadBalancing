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
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

        # Initialize weights for better training
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
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


# Optimized training parameters
BATCH_SIZE = 32  # Reduced from 64 to train with fewer samples
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
LEARNING_RATE = 3e-4  # Adjusted for better convergence
MEMORY_SIZE = 50000  # Increased for better experience diversity
MIN_MEMORY_SIZE = 100  # Reduced from 1000 to match your environment's scale
TARGET_UPDATE = 10
UPDATE_FREQ = 2  # More frequent updates

# Create the environment
env = NetworkSimEnv(
    render_mode="human",
    num_controllers=4,
    num_switches=26,
    max_rate=1000,
    gc_ip="localhost",
    gc_port="8000",
    step_time=1,
)

# Initialize networks
input_dim = env.m * env.n
output_dim = env.action_space.n

policy_net = DQN(input_dim, output_dim).to(device)
target_net = DQN(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())

# Use Adam optimizer with AMSGrad for better stability
optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayBuffer(MEMORY_SIZE, input_dim)
epsilon = EPSILON_START


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

    # Convert to tensors efficiently
    states = torch.FloatTensor(states).to(device)
    actions = torch.LongTensor(actions - 1).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    next_states = torch.FloatTensor(next_states).to(device)
    dones = torch.FloatTensor(dones).to(device)

    # Compute Q values in a single forward pass
    current_q_values = policy_net(states).gather(1, actions.unsqueeze(1))
    with torch.no_grad():
        next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)

    # Compute Huber loss for better stability
    loss = nn.SmoothL1Loss()(current_q_values.squeeze(), expected_q_values)

    optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
    loss.backward()
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item()


# Training loop with performance monitoring
num_episodes = 100
total_steps = 0
running_reward = 0
print_every = 1

for episode in range(num_episodes):
    state, _ = env.reset()
    episode_reward = 0
    episode_loss = 0
    episode_steps = 0

    for step in range(100):  # Max steps per episode
        total_steps += 1
        episode_steps += 1

        action = select_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        print(f"State: {state}")
        print(f"Reward: {reward}")

        memory.push(state, action, reward, next_state, done)
        state = next_state
        episode_reward += reward

        if total_steps % UPDATE_FREQ == 0:
            loss = optimize_model()
            if loss is not None:
                episode_loss += loss
                print(
                    f"Step {step}, Loss: {loss:.4f}, Buffer size: {len(memory)}/{MIN_MEMORY_SIZE}"
                )
            else:
                print(f"No training yet. Buffer size: {len(memory)}/{MIN_MEMORY_SIZE}")

        if done or truncated:
            break

    # Update target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    # Decay epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    # Update running reward
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    if (episode + 1) % print_every == 0:
        avg_loss = episode_loss / episode_steps if episode_steps > 0 else 0
        print(f"Episode {episode + 1}/{num_episodes}")
        print(f"Steps: {episode_steps}, Total Steps: {total_steps}")
        print(f"Reward: {episode_reward:.2f}, Running Reward: {running_reward:.2f}")
        print(f"Average Loss: {avg_loss:.4f}, Epsilon: {epsilon:.3f}")
        print("-" * 50)

    # Save the model periodically
    if (episode + 1) % 100 == 0:
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

env.close()
