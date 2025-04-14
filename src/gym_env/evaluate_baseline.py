import argparse
import numpy as np
from train import NetworkSimEnv
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

def evaluate_no_migration(env, num_episodes=10, render=False):
    """
    Evaluate the network performance using the default topology mapping.
    Always choose actions that keep switches with their original controllers.
    """
    episode_rewards = []
    episode_lengths = []
    balancing_rates = []  # Store D_t values
    load_ratios = []  # Store B values

    # Define default mapping based on Janos US topology
    default_mapping = {
        # Domain 1 (Controller 1)
        1: 1,
        2: 1,
        3: 1,
        4: 1,
        5: 1,
        6: 1,
        7: 1,
        # Domain 2 (Controller 2)
        8: 2,
        9: 2,
        10: 2,
        11: 2,
        12: 2,
        13: 2,
        # Domain 3 (Controller 3)
        14: 3,
        15: 3,
        16: 3,
        17: 3,
        18: 3,
        19: 3,
        # Domain 4 (Controller 4)
        20: 4,
        21: 4,
        22: 4,
        23: 4,
        24: 4,
        25: 4,
        26: 4,
    }

    print("\nDefault topology switch-to-controller mapping:")
    for controller in range(1, env.m + 1):
        switches = [s for s, c in default_mapping.items() if c == controller]
        print(f"Controller {controller}: Switches {sorted(switches)}")

    for episode in tqdm(
        range(num_episodes), desc="Evaluating Default Mapping Strategy"
    ):
        state, _ = env.reset()
        episode_reward = 0
        steps = 0
        episode_d_t = []
        episode_load_ratios = []

        # Wait for warm-up
        while not np.any(state != 0):
            state, _, _, _, _ = env.step(env.action_space.sample())

        # Run episode
        done = False
        while not done and steps < 200:  # Max 200 steps per episode
            # Choose action that maintains default mapping
            switch_num = (steps % env.n) + 1  # Cycle through switches 1 to n
            current_controller = default_mapping[switch_num]

            # Calculate action that keeps switch with its default controller
            action = (current_controller - 1) * env.n + switch_num

            # Take step in environment
            next_state, reward, done, truncated, info = env.step(action)
            print(f"Next state: {next_state}")

            # Calculate and store metrics
            L = np.sum(state, axis=1)  # Load for each controller
            B = L / env.capacities  # Load ratios
            B_bar = np.sum(B) / env.m  # Average load

            if B_bar != 0:
                D_t = np.sqrt(np.sum((B - B_bar) ** 2) / env.m) / B_bar
            else:
                D_t = 1

            episode_d_t.append(D_t)
            episode_load_ratios.append(B)

            state = next_state
            episode_reward += reward
            steps += 1

            if render:
                env.render()

            if done or truncated:
                break

        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        balancing_rates.append(episode_d_t)
        load_ratios.append(episode_load_ratios)

    return {
        "rewards": episode_rewards,
        "lengths": episode_lengths,
        "balancing_rates": balancing_rates,
        "load_ratios": load_ratios,
        "default_mapping": default_mapping,
    }


def plot_baseline_results(results, timestamp):
    """
    Plot evaluation metrics for baseline no-migration strategy
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Plot episode rewards
    ax1.plot(results["rewards"])
    ax1.set_title("Episode Rewards (No Migration)")
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

    plt.suptitle("Baseline Evaluation Results (No Migration Strategy)")
    plt.tight_layout()
    plt.savefig(f"baseline_evaluation_results_{timestamp}.png")
    plt.close()


def save_results(results, timestamp):
    """
    Save the evaluation results for later comparison
    """
    filename = f"baseline_results_{timestamp}.npz"
    np.savez(
        filename,
        rewards=results["rewards"],
        lengths=results["lengths"],
        balancing_rates=results["balancing_rates"],
        load_ratios=results["load_ratios"],
        default_mapping=results["default_mapping"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline no-migration strategy"
    )
    parser.add_argument(
        "--episodes", type=int, default=10, help="Number of evaluation episodes"
    )
    parser.add_argument("--render", action="store_true", help="Render the environment")
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

    # Run evaluation
    results = evaluate_no_migration(env, num_episodes=args.episodes, render=args.render)

    # Plot and save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_baseline_results(results, timestamp)
    save_results(results, timestamp)

    # Print summary statistics
    print("\nBaseline Evaluation Results (No Migration):")
    print(
        f"Average Reward: {np.mean(results['rewards']):.2f} ± {np.std(results['rewards']):.2f}"
    )
    print(
        f"Average Episode Length: {np.mean(results['lengths']):.2f} ± {np.std(results['lengths']):.2f}"
    )
    print(
        f"Average Balancing Rate: {np.mean([np.mean(d_t) for d_t in results['balancing_rates']]):.4f}"
    )

    # Calculate and print additional metrics
    avg_load_ratios = np.mean(
        [np.mean(ratios, axis=0) for ratios in results["load_ratios"]], axis=0
    )
    print("\nAverage Load Distribution across Controllers:")
    for i, load in enumerate(avg_load_ratios, 1):
        print(f"Controller {i}: {load:.4f}")

    print(f"\nLoad Imbalance (std dev): {np.std(avg_load_ratios):.4f}")

    # Close environment
    # env.close()


if __name__ == "__main__":
    main()
