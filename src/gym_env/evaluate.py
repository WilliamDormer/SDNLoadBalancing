import argparse
import torch
import numpy as np
from train import DQN, NetworkSimEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import json
from datetime import datetime

# Set the default device
device = (
    torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
)
print(f"Using device: {device}")


def load_model(checkpoint_path, input_dim, output_dim):
    """
    Load a trained model from checkpoint
    """
    # Initialize the DQN with the same architecture
    model = DQN(input_dim, output_dim).to(device)

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Set to evaluation mode
    model.eval()

    print(f"Loaded model from episode {checkpoint['episode'] + 1}")
    return model, checkpoint["episode"]


def evaluate(model, env, num_episodes=10, render=False):
    """
    Evaluate the model's performance with non-migration warm-up strategy
    """
    episode_rewards = []
    episode_lengths = []
    balancing_rates = []  # Store D_t values
    load_ratios = []  # Store B values
    actions_taken = []  # Store actions for analysis
    switch_migrations = []  # Store switch migration events

    with torch.no_grad():
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            state, _ = env.reset()
            episode_reward = 0
            steps = 0
            episode_d_t = []
            episode_load_ratios = []
            episode_actions = []
            episode_migrations = []

            prev_config = [list(c) for c in env.switches_by_controller]

            # Wait for warm-up
            while not np.any(state != 0):
                print(f"Warm-up")
                state, _, _, _, _ = env.step(0)


            # Main evaluation loop
            done = False
            while not done and steps < 200:  # Max 200 steps per episode
                # Calculate current loads and metrics
                L = np.sum(state, axis=1)  # Load for each controller
                B = L / env.capacities  # Load ratios
                B_bar = np.sum(B) / env.m  # Average load

                if B_bar != 0:
                    D_t = np.sqrt(np.sum((B - B_bar) ** 2) / env.m) / B_bar
                else:
                    D_t = 1

                # Get action from model
                state_tensor = (
                    torch.FloatTensor(state.flatten()).unsqueeze(0).to(device)
                )
                action = (
                    model(state_tensor).max(1)[1].item() + 1
                )  # +1 because 0 is no-op

                # Store action
                episode_actions.append(action)

                # Take step in environment
                next_state, reward, done, truncated, info = env.step(action)

                # Record migration with load information if configuration changed
                current_config = [list(c) for c in env.switches_by_controller]
                if current_config != prev_config:
                    migration_info = {
                        "step": steps,
                        "action": action,
                        "config": current_config.copy(),
                        "loads": {
                            "absolute_loads": L.tolist(),  # Raw loads
                            "load_ratios": B.tolist(),  # Load ratios
                            "average_load": float(B_bar),  # Average load
                            "imbalance": float(D_t),  # Current D_t value
                            "capacities": env.capacities.tolist(),  # Controller capacities
                        },
                    }
                    episode_migrations.append(migration_info)
                prev_config = current_config

                episode_d_t.append(D_t)
                episode_load_ratios.append(B)

                state = next_state
                episode_reward += reward
                steps += 1

                if render:
                    env.render()

                if done or truncated:
                    break

            # Store episode results
            episode_rewards.append(episode_reward)
            episode_lengths.append(steps)
            balancing_rates.append(episode_d_t)
            load_ratios.append(episode_load_ratios)
            actions_taken.append(episode_actions)
            switch_migrations.append(episode_migrations)

            print(f"Episode {episode + 1} completed:")
            print(f"  Steps: {steps}")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Migrations: {len(episode_migrations)}")
            print(f"  Final D_t: {D_t:.4f}")

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "balancing_rates": balancing_rates,
        "load_ratios": load_ratios,
        "actions": actions_taken,
        "migrations": switch_migrations,
    }


def plot_evaluation_results(results, episode_num, timestamp):
    """
    Plot evaluation metrics
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot episode rewards
    ax1.plot(results["rewards"])
    ax1.set_title("Episode Rewards")
    ax1.set_xlabel("Evaluation Episode")
    ax1.set_ylabel("Total Reward")

    # Plot episode lengths
    ax2.plot(results["lengths"])
    ax2.set_title("Episode Lengths")
    ax2.set_xlabel("Evaluation Episode")
    ax2.set_ylabel("Steps")

    # Plot average balancing rates over time
    avg_d_t = [np.mean(d_t) for d_t in results["balancing_rates"]]
    ax3.plot(avg_d_t)
    ax3.set_title("Average Balancing Rate (D_t)")
    ax3.set_xlabel("Evaluation Episode")
    ax3.set_ylabel("D_t")

    # Plot load distribution
    avg_load_ratios = np.mean(
        [np.mean(ratios, axis=0) for ratios in results["load_ratios"]], axis=0
    )
    controllers = range(1, len(avg_load_ratios) + 1)
    ax4.bar(controllers, avg_load_ratios)
    ax4.set_title("Average Load Distribution")
    ax4.set_xlabel("Controller")
    ax4.set_ylabel("Average Load Ratio")

    plt.suptitle(f"Evaluation Results for Model at Episode {episode_num}")
    plt.tight_layout()
    plt.savefig(f"evaluation_results_episode_{episode_num}_{timestamp}.png")
    plt.close()


def save_evaluation_results(results, episode_num, filename=None, timestamp=None):
    """
    Save evaluation results to NPZ file with proper handling of irregular arrays
    """
    if filename is None:
        filename = f"dqn_evaluation_episode_{episode_num}_{timestamp}.npz"

    # Convert regular arrays
    rewards = np.array(results["rewards"])
    lengths = np.array(results["lengths"])

    # Handle irregular arrays by padding or converting to object arrays
    balancing_rates = np.array(results["balancing_rates"], dtype=object)
    load_ratios = np.array(results["load_ratios"], dtype=object)
    actions = np.array(results["actions"], dtype=object)

    # Save migrations separately as JSON for better handling of irregular structure
    migration_file = filename.replace(".npz", "_migrations.json")
    with open(migration_file, "w") as f:
        json.dump(results["migrations"], f, default=lambda x: str(x))

    # Save the regular arrays
    np.savez(
        filename,
        rewards=rewards,
        lengths=lengths,
        balancing_rates=balancing_rates,
        load_ratios=load_ratios,
        actions=actions,
        episode_num=episode_num,
    )

    print(f"Saved evaluation results to {filename}")
    print(f"Saved migration data to {migration_file}")


def load_evaluation_results(filename):
    """
    Load evaluation results from NPZ file and JSON
    """
    # Load main results
    with np.load(filename, allow_pickle=True) as data:
        results = dict(data)

    # Load migrations data
    migration_file = filename.replace(".npz", "_migrations.json")
    if os.path.exists(migration_file):
        with open(migration_file, "r") as f:
            results["migrations"] = json.load(f)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained DQN model")
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
    parser.add_argument("--output", type=str, help="Output NPZ file name (optional)")
    args = parser.parse_args()

    # Create environment
    env = NetworkSimEnv(
        render_mode="human" if args.render else "no",
        num_controllers=4,
        num_switches=26,
        max_rate=1000,
        gc_ip="localhost",
        gc_port="8000",
        step_time=0.5,
    )

    # Load model
    input_dim = env.m * env.n
    output_dim = env.action_space.n
    model, episode_num = load_model(args.checkpoint, input_dim, output_dim)

    # Run evaluation
    results = evaluate(model, env, num_episodes=args.episodes, render=args.render)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        # Save results
        save_evaluation_results(results, episode_num, args.output, timestamp)
    except Exception as e:
        print(f"Error saving results: {e}")
        # Save as pickle as fallback
        import pickle

        fallback_file = "evaluation_results_fallback.pkl"
        with open(fallback_file, "wb") as f:
            pickle.dump({"results": results, "episode_num": episode_num}, f)
        print(f"Saved results to fallback file: {fallback_file}")

    # Plot and save results
    plot_evaluation_results(results, episode_num, timestamp)

    # Print summary statistics
    print("\nEvaluation Results:")
    print(
        f"Average Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}"
    )
    print(
        f"Average Episode Length: {np.mean(results['lengths']):.2f} ± {np.std(results['lengths']):.2f}"
    )
    print(
        f"Average Balancing Rate: {np.mean([np.mean(d_t) for d_t in results['balancing_rates']]):.4f}"
    )

    # Calculate and print migration statistics
    total_migrations = sum(
        len(episode_migrations) for episode_migrations in results["migrations"]
    )
    avg_migrations = total_migrations / args.episodes
    print(f"Average Migrations per Episode: {avg_migrations:.2f}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
