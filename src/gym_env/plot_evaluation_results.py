import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os
import json


def plot_episode_loads(episode_data, episode_num, save_dir, show_plot=False):
    """
    Create detailed load plots for a single episode
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: Controller loads over time
    time_steps = range(len(episode_data))
    num_controllers = episode_data.shape[1]

    for controller in range(num_controllers):
        ax1.plot(
            time_steps,
            episode_data[:, controller],
            label=f"Controller {controller + 1}",
            linewidth=2,
        )

    # Add mean and std dev to the plot
    mean_loads = np.mean(episode_data, axis=1)

    # Calculate std dev safely
    try:
        std_loads = np.zeros_like(mean_loads)  # Initialize with zeros
        if (
            episode_data.shape[1] > 1
        ):  # Only calculate std if we have more than one controller
            var = np.var(episode_data, axis=1)
            std_loads = np.sqrt(
                var + 1e-10
            )  # Add small constant to avoid sqrt of negative numbers
    except Exception as e:
        print(f"Warning: Could not calculate standard deviation: {e}")
        std_loads = np.zeros_like(mean_loads)

    ax1.plot(time_steps, mean_loads, "k--", label="Mean Load", linewidth=2)
    ax1.fill_between(
        time_steps,
        mean_loads - std_loads,
        mean_loads + std_loads,
        alpha=0.2,
        color="gray",
        label="±1 Std Dev",
    )

    ax1.set_title(f"Controller Loads - Episode {episode_num}")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Load Ratio")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Box plot of load distributions
    controller_loads = [episode_data[:, i] for i in range(num_controllers)]
    ax2.boxplot(
        controller_loads,
        labels=[f"Controller {i+1}" for i in range(num_controllers)],
    )
    ax2.set_title(f"Load Distribution by Controller - Episode {episode_num}")
    ax2.set_ylabel("Load Ratio")
    ax2.grid(True)

    # Add episode statistics
    try:
        mean_load = np.mean(episode_data)
        std_dev = np.std(episode_data)
        load_imbalance = np.std(np.mean(episode_data, axis=0))
        max_load = np.max(episode_data)
    except Exception as e:
        print(f"Warning: Could not calculate some statistics: {e}")
        mean_load = std_dev = load_imbalance = max_load = 0

    stats_text = f"Episode Statistics:\n"
    stats_text += f"Mean Load: {mean_load:.3f}\n"
    stats_text += f"Std Dev: {std_dev:.3f}\n"
    stats_text += f"Load Imbalance: {load_imbalance:.3f}\n"
    stats_text += f"Max Load: {max_load:.3f}"

    # Add text box with statistics
    ax2.text(
        1.15,
        0.5,
        stats_text,
        transform=ax2.transAxes,
        bbox=dict(facecolor="white", alpha=0.8),
        verticalalignment="center",
    )

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"episode_{episode_num}_loads.png"))
    if show_plot:
        plt.show()
    plt.close()


def plot_evaluation_results(npz_file, save_dir="evaluation_plots", show_plot=False):
    """
    Load NPZ file and create detailed plots for DQN evaluation results
    """
    # Load the results
    results = np.load(npz_file, allow_pickle=True)

    # Extract data
    rewards = results["rewards"]
    lengths = results["lengths"]
    balancing_rates = results["balancing_rates"]
    load_ratios = results["load_ratios"]
    actions = results["actions"]
    episode_num = results["episode_num"]

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create episode-specific directory
    episodes_dir = os.path.join(save_dir, "episode_plots")
    if not os.path.exists(episodes_dir):
        os.makedirs(episodes_dir)

    # Plot individual episode loads
    print("Generating episode plots...")
    for i, episode_loads in enumerate(tqdm(load_ratios)):
        try:
            episode_loads = np.array(episode_loads, dtype=np.float64)
            if len(episode_loads.shape) != 2:
                print(f"Warning: Skipping episode {i+1} due to invalid data shape")
                continue
            plot_episode_loads(episode_loads, i + 1, episodes_dir, show_plot)
        except Exception as e:
            print(f"Error processing episode {i+1}: {e}")
            continue

    # Plot summary metrics
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot rewards and lengths
    episodes = range(1, len(rewards) + 1)
    ax1.plot(episodes, rewards, "b-", label="Episode Reward", marker="o")
    ax1.set_title("Episode Performance Metrics")
    ax1.set_xlabel("Evaluation Episode")
    ax1.set_ylabel("Reward", color="b")
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(episodes, lengths, "r-", label="Episode Length", marker="s")
    ax1_twin.set_ylabel("Length", color="r")
    ax1_twin.tick_params(axis="y", labelcolor="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # Plot balancing rates
    mean_balancing = [np.mean(rates) for rates in balancing_rates]
    std_balancing = [
        np.std(rates) if len(rates) > 1 else 0 for rates in balancing_rates
    ]

    ax2.plot(episodes, mean_balancing, "g-", label="Mean Balancing Rate", marker="o")
    ax2.fill_between(
        episodes,
        np.array(mean_balancing) - np.array(std_balancing),
        np.array(mean_balancing) + np.array(std_balancing),
        alpha=0.2,
        color="g",
        label="±1 Std Dev",
    )
    ax2.set_title("Balancing Rate Evolution")
    ax2.set_xlabel("Evaluation Episode")
    ax2.set_ylabel("Balancing Rate (D_t)")
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_metrics.png"))
    if show_plot:
        plt.show()
    plt.close()

    # Print summary statistics
    print("\nEvaluation Summary:")
    print(f"Model checkpoint episode: {episode_num}")
    print(f"Number of evaluation episodes: {len(rewards)}")
    print(f"Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"Average episode length: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}")
    print(
        f"Average balancing rate: {np.mean(mean_balancing):.4f} ± {np.std(mean_balancing):.4f}"
    )


def main():
    parser = argparse.ArgumentParser(description="Analyze DQN evaluation results")
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="Path to NPZ file containing evaluation results",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots while generating"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_plots",
        help="Directory to save plots",
    )
    args = parser.parse_args()

    plot_evaluation_results(args.file, args.output_dir, args.show)


if __name__ == "__main__":
    main()
