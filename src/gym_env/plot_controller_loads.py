import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import os


def plot_episode_loads(npz_file, save_dir="episode_plots", show_plot=False):
    """
    Load NPZ file and create separate plots for each episode's controller loads
    """
    # Load the results
    results = np.load(npz_file)
    load_ratios = results["load_ratios"]

    # Get dimensions
    num_episodes = len(load_ratios)
    steps_per_episode = len(load_ratios[0])
    num_controllers = len(load_ratios[0][0])

    print(f"Loaded data with:")
    print(f"Number of episodes: {num_episodes}")
    print(f"Steps per episode: {steps_per_episode}")
    print(f"Number of controllers: {num_controllers}")

    # Create save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot each episode
    for episode in tqdm(range(num_episodes), desc="Plotting episodes"):
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

        # Plot 1: Controller loads over time
        episode_data = load_ratios[episode]
        time_steps = range(steps_per_episode)

        for controller in range(num_controllers):
            ax1.plot(
                time_steps,
                episode_data[:, controller],
                label=f"Controller {controller + 1}",
                linewidth=2,
            )

        ax1.set_title(f"Controller Loads - Episode {episode + 1}")
        ax1.set_xlabel("Time Step")
        ax1.set_ylabel("Load Ratio")
        ax1.legend()
        ax1.grid(True)

        # Add mean and std dev to the plot
        mean_loads = np.mean(episode_data, axis=1)
        std_loads = np.std(episode_data, axis=1)
        ax1.plot(time_steps, mean_loads, "k--", label="Mean Load", linewidth=2)
        ax1.fill_between(
            time_steps,
            mean_loads - std_loads,
            mean_loads + std_loads,
            alpha=0.2,
            color="gray",
            label="±1 Std Dev",
        )

        # Plot 2: Box plot of load distributions for this episode
        controller_loads = [episode_data[:, i] for i in range(num_controllers)]
        ax2.boxplot(
            controller_loads,
            labels=[f"Controller {i+1}" for i in range(num_controllers)],
        )
        ax2.set_title(f"Load Distribution by Controller - Episode {episode + 1}")
        ax2.set_ylabel("Load Ratio")
        ax2.grid(True)

        # Add episode statistics
        stats_text = f"Episode Statistics:\n"
        stats_text += f"Mean Load: {np.mean(episode_data):.3f}\n"
        stats_text += f"Std Dev: {np.std(episode_data):.3f}\n"
        stats_text += f"Load Imbalance: {np.std(np.mean(episode_data, axis=0)):.3f}\n"
        stats_text += f"Max Load: {np.max(episode_data):.3f}"

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

        # Save the plot
        plt.savefig(os.path.join(save_dir, f"episode_{episode+1}_loads.png"))

        if show_plot:
            plt.show()
        plt.close()

    # Create summary plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

    # Plot 1: Average controller loads across all episodes
    avg_loads = np.mean(
        [np.array(episode_loads) for episode_loads in load_ratios], axis=0
    )
    std_loads = np.std(
        [np.array(episode_loads) for episode_loads in load_ratios], axis=0
    )
    time_steps = range(steps_per_episode)

    for controller in range(num_controllers):
        ax1.plot(
            time_steps,
            avg_loads[:, controller],
            label=f"Controller {controller + 1}",
            linewidth=2,
        )
        ax1.fill_between(
            time_steps,
            avg_loads[:, controller] - std_loads[:, controller],
            avg_loads[:, controller] + std_loads[:, controller],
            alpha=0.2,
        )

    ax1.set_title("Average Controller Loads Across All Episodes")
    ax1.set_xlabel("Time Step")
    ax1.set_ylabel("Load Ratio")
    ax1.legend()
    ax1.grid(True)

    # Plot 2: Evolution of load imbalance
    imbalances = [np.std(np.mean(episode, axis=0)) for episode in load_ratios]
    ax2.plot(range(1, num_episodes + 1), imbalances, marker="o")
    ax2.set_title("Load Imbalance Evolution")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Load Imbalance (Std Dev)")
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "summary_analysis.png"))

    if show_plot:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze controller loads from NPZ file"
    )
    parser.add_argument(
        "--file", type=str, required=True, help="Path to NPZ file containing results"
    )
    parser.add_argument(
        "--show", action="store_true", help="Display plots while generating"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="episode_plots",
        help="Directory to save episode plots",
    )
    args = parser.parse_args()

    plot_episode_loads(args.file, args.output_dir, args.show)


if __name__ == "__main__":
    main()
