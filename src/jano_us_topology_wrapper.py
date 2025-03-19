#!/usr/bin/env python3

from topology.jano_us_topology import JanosUSTopology
from flask import Flask, jsonify, request
from ryu.lib import hub
import argparse
import numpy as np
import time
import urllib.request
import json
import os

class JanosUSTopologyWrapper:
    """
    A wrapper class that adds a REST API to the JanosUSTopology,
    allowing the global controller to send reset requests and get network state.
    Designed to work with OpenAI Gym environments.
    """

    def __init__(self, args):
        """
        Initialize the wrapper with a Flask server and the topology.

        Args:
            args: Command line arguments for JanosUSTopology
        """
        # Store controller info first
        self.global_controller_ip = args.global_controller_ip
        self.global_controller_port = args.global_controller_port

        # Store network parameters
        self.num_switches = 26  # Total number of switches in topology
        self.num_domains = 4  # Number of domains/controllers

        # Initialize Flask app
        self.app = Flask(__name__)

        # clear mininet
        os.system("sudo mn -c")

        # Initialize topology
        self.topology = JanosUSTopology(args)

        # Track current state
        self.current_state = np.zeros((self.num_domains, self.num_switches))
        self.start_time = time.time()

        # Register routes
        @self.app.route("/reset", methods=["POST"])
        def reset_topology():
            """Handle reset requests and return initial state"""
            try:
                # Reset the topology
                self.topology.reset()

                # Wait for network to stabilize
                while self.topology.is_resetting:
                    time.sleep(1)
                print("Network reset complete")

                # Get initial state
                initial_state = self._get_network_state()
                self.current_state = initial_state
                self.start_time = time.time()

                return (
                    jsonify(
                        {
                            "message": "Topology reset successful",
                            "state": initial_state,
                            "start_time": self.start_time,
                        }
                    ),
                    200,
                )
            except Exception as e:
                return jsonify({"error": f"Reset failed: {str(e)}"}), 500

        @self.app.route("/step", methods=["POST"])
        def step():
            """
            Handle step requests from the gym environment.
            Generates traffic for one step and returns the new state.
            """
            try:
                # Get action from request body (if any)
                action = request.json.get("action", None)

                if action is not None:
                    # Prepare the data
                    data = json.dumps(
                        {"target_controller": action[0], "switch": action[1]}
                    ).encode("utf-8")

                    # Create request with proper headers and data
                    # req = urllib.request.Request(
                    #     f"http://{self.global_controller_ip}:{self.global_controller_port}/migrate",
                    #     data=data,
                    #     headers={"Content-Type": "application/json"},
                    # )

                    # with urllib.request.urlopen(req) as response:
                    #     response_text = response.read().decode("utf-8")
                    #     response_code = response.getcode()

                    #     if response_code != 200:
                    #         return (
                    #             jsonify(
                    #                 {
                    #                     "error": f"Failed to migrate switch: {response_text}"
                    #                 }
                    #             ),
                    #             500,
                    #         )

                # Generate traffic for one step
                self.topology.generate_flows(time.time() + self.topology.flow_duration)

                # Wait for flows to be established
                time.sleep(self.topology.flow_duration - 2)

                # Get new state
                new_state = self._get_network_state()
                self.current_state = new_state

                # Calculate reward (placeholder - implement your reward function)
                # reward = self._calculate_reward(new_state)
                reward = 0
                return (
                    jsonify(
                        {
                            "state": new_state,
                            "reward": reward,
                            "terminated": False,
                            "truncated": False,
                            "info": {},
                            "done": False,
                        }
                    ),
                    200,
                )
            except Exception as e:
                return jsonify({"error": f"Step failed: {str(e)}"}), 500

        # Start Flask server in a separate thread
        self.flask_port = 9000  # Different from global controller's port
        self.flask_thread = hub.spawn(self.run_flask)

    def _get_network_state(self):
        """
        Get the current state of the network.
        Returns a state representation suitable for RL.
        """
        req = urllib.request.Request(
            f"http://{self.global_controller_ip}:{self.global_controller_port}/state",
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req) as response:
            response_text = response.read().decode("utf-8")
            response_code = response.getcode()

            if response_code != 200:
                return jsonify({"error": f"Failed to get state: {response_text}"}), 500

            state = json.loads(response_text)["state"]

        return state

    def _calculate_reward(self, state):
        """
        Calculate the reward based on the current state.

        Args:
            state: The current state matrix (num_domains x num_switches)

        Returns:
            float: The calculated reward
        """
        # TODO: Implement actual reward function
        # Example reward: negative sum of squared differences from mean load
        domain_loads = state.sum(axis=1)  # Sum loads for each domain
        mean_load = domain_loads.mean()
        load_variance = ((domain_loads - mean_load) ** 2).sum()

        # Return negative variance (we want to minimize variance)
        return -load_variance

    def run_flask(self):
        """Run the Flask server"""
        self.app.run(host="0.0.0.0", port=self.flask_port)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_ip", type=str, default="192.168.2.33")
    parser.add_argument("--base_lambda", type=float, default=5.0)
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--period", type=float, default=60.0)
    parser.add_argument("--max_bw", type=str, default="10M")
    parser.add_argument("--flow_duration", type=int, default=10)
    parser.add_argument("--global_controller_ip", type=str, default="192.168.2.33")
    parser.add_argument("--global_controller_port", type=int, default=8000)
    args = parser.parse_args()

    # Create and start the wrapper
    wrapper = JanosUSTopologyWrapper(args)

    # Keep the main thread running
    try:
        hub.sleep(float("inf"))
    except KeyboardInterrupt:
        print("\nShutting down topology wrapper...")
