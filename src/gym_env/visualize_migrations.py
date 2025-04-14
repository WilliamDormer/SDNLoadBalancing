import json
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np
from pathlib import Path


def create_network_graph(config, num_controllers=4):
    """Create a NetworkX graph from the switch configuration"""
    G = nx.Graph()

    # Add controller nodes
    controller_pos = {
        0: (-1, 1),  # Top left
        1: (1, 1),  # Top right
        2: (-1, -1),  # Bottom left
        3: (1, -1),  # Bottom right
    }

    for i in range(num_controllers):
        G.add_node(f"C{i+1}", pos=controller_pos[i], node_type="controller")

    # Add switch nodes and connect them to their controllers
    switch_pos = {}
    for controller_idx, switches in enumerate(config):
        controller = f"C{controller_idx+1}"
        angle_step = 2 * np.pi / (len(switches) + 1) if switches else 0
        radius = 0.5

        for i, switch in enumerate(switches):
            switch_name = f"S{switch}"
            if switch_name not in G:
                # Calculate position in a circle around the controller
                angle = i * angle_step
                cx, cy = controller_pos[controller_idx]
                x = cx + radius * np.cos(angle)
                y = cy + radius * np.sin(angle)
                switch_pos[switch_name] = (x, y)

                G.add_node(switch_name, pos=switch_pos[switch_name], node_type="switch")
                G.add_edge(controller, switch_name)

    return G


def update(frame, ax, migrations, num_controllers):
    """Update function for animation"""
    ax.clear()

    if frame < len(migrations):
        migration = migrations[frame]
        config = migration["config"]
        step = migration["step"]
        action = migration["action"]
        loads = migration["loads"]

        G = create_network_graph(config, num_controllers)
        pos = nx.get_node_attributes(G, "pos")

        # Draw the graph
        controller_nodes = [
            n for n in G.nodes() if G.nodes[n]["node_type"] == "controller"
        ]
        load_ratios = loads["load_ratios"]

        # Fixed scale from 0 to 1.6 for consistent coloring
        norm = plt.Normalize(0, 1.6)  # Fixed maximum at 1.6
        cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green colormap, reversed

        # Draw controllers with colors based on load
        for i, node in enumerate(controller_nodes):
            load = load_ratios[i]
            color = cmap(norm(load))
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=[node],
                node_color=[color],
                node_size=1000,
                node_shape="s",
            )

        # Draw switches
        switch_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "switch"]
        nx.draw_networkx_nodes(
            G, pos, nodelist=switch_nodes, node_color="lightblue", node_size=500
        )

        # Draw edges
        nx.draw_networkx_edges(G, pos)

        # Add load labels to controllers
        controller_labels = {}
        for i, node in enumerate(controller_nodes):
            load = load_ratios[i]
            abs_load = loads["absolute_loads"][i]
            controller_labels[node] = f"{load:.2f}\n({abs_load:.0f})"
        nx.draw_networkx_labels(G, pos, controller_labels)

        # Add switch labels
        switch_labels = {node: node.replace("S", "") for node in switch_nodes}
        nx.draw_networkx_labels(G, pos, switch_labels)

        # Add title with step, action, and imbalance information
        title = f"Step: {step}, Action: {action}\n"
        title += f"Imbalance: {loads['imbalance']:.3f}\n"
        title += f"Avg Load: {loads['average_load']:.3f}"
        ax.set_title(title)

        # Add colorbar with fixed scale
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax, label="Load Ratio")
        cbar.set_ticks([0, 0.4, 0.8, 1.2, 1.6])  # Set specific tick points

    ax.set_xlim([-1.8, 1.8])
    ax.set_ylim([-1.8, 1.8])
    ax.axis("equal")


def create_migration_animation(migration_file, output_file=None, fps=2):
    """Create an animation from the migration data"""
    # Load migration data
    with open(migration_file, "r") as f:
        episodes = json.load(f)

    # Process the first episode
    migrations = episodes[0]
    num_controllers = 4

    # Create figure with more space for colorbar
    fig, ax = plt.subplots(figsize=(12, 10))
    plt.subplots_adjust(right=0.85)  # Make room for colorbar

    # Create colorbar once
    norm = plt.Normalize(0, 1.6)
    cmap = plt.cm.RdYlGn_r
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cbar = plt.colorbar(sm, ax=ax, label="Load Ratio")
    cbar.set_ticks([0, 0.4, 0.8, 1.2, 1.6])

    def update(frame):
        """Update function for animation"""
        ax.clear()

        if frame < len(migrations):
            migration = migrations[frame]
            config = migration["config"]
            step = migration["step"]
            action = migration["action"]
            loads = migration["loads"]

            G = create_network_graph(config, num_controllers)
            pos = nx.get_node_attributes(G, "pos")

            # Draw controllers with colors based on load
            controller_nodes = [
                n for n in G.nodes() if G.nodes[n]["node_type"] == "controller"
            ]
            load_ratios = loads["load_ratios"]

            for i, node in enumerate(controller_nodes):
                load = load_ratios[i]
                color = cmap(norm(load))
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    nodelist=[node],
                    node_color=[color],
                    node_size=1000,
                    node_shape="s",
                )

            # Draw switches
            switch_nodes = [n for n in G.nodes() if G.nodes[n]["node_type"] == "switch"]
            nx.draw_networkx_nodes(
                G, pos, nodelist=switch_nodes, node_color="lightblue", node_size=500
            )

            # Draw edges
            nx.draw_networkx_edges(G, pos)

            # Add load labels to controllers
            controller_labels = {}
            for i, node in enumerate(controller_nodes):
                load = load_ratios[i]
                abs_load = loads["absolute_loads"][i]
                controller_labels[node] = f"{load:.2f}\n({abs_load:.0f})"
            nx.draw_networkx_labels(G, pos, controller_labels)

            # Add switch labels
            switch_labels = {node: node.replace("S", "") for node in switch_nodes}
            nx.draw_networkx_labels(G, pos, switch_labels)

            # Add title with step, action, and imbalance information
            title = f"Step: {step}, Action: {action}\n"
            title += f"Imbalance: {loads['imbalance']:.3f}\n"
            title += f"Avg Load: {loads['average_load']:.3f}"
            ax.set_title(title)

        ax.set_xlim([-1.8, 1.8])
        ax.set_ylim([-1.8, 1.8])
        ax.axis("equal")

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=len(migrations), interval=1000 / fps, repeat=True
    )

    # Save animation with higher DPI and better quality
    if output_file is None:
        output_file = Path(migration_file).stem + "_animation.gif"

    writer = PillowWriter(fps=fps)
    anim.save(output_file, writer=writer, dpi=150)
    plt.close()

    print(f"Animation saved as {output_file}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Create animation from migration data")
    parser.add_argument(
        "--file", type=str, required=True, help="Path to migration JSON file"
    )
    parser.add_argument("--output", type=str, default=None, help="Output GIF file path")
    parser.add_argument(
        "--fps", type=int, default=2, help="Frames per second for the animation"
    )

    args = parser.parse_args()
    create_migration_animation(args.file, args.output, args.fps)


if __name__ == "__main__":
    main()
