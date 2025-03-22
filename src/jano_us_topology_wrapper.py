#!/usr/bin/env python3

from topology.jano_us_topology import JanosUSTopology
from flask import Flask, jsonify, request
from ryu.lib import hub
import argparse
import numpy as np
import time
import threading  # Use threading instead of hub


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
        self.flask_port = 9000 # TODO move this to a config file
        # Initialize topology
        self.topology = JanosUSTopology(args)

        # Track current state
        self.current_state = np.zeros((self.num_domains, self.num_switches))
        self.start_time = time.time()

        # Register routes
        @self.app.route("/reset_topology", methods=["POST"])
        def reset_topology():
            """Handle reset requests and return initial state"""
            try:
                # kill the thread running the simulation
                if self.sim_thread and self.sim_thread.is_alive():
                    self.sim_thread.stop()
                    time.sleep(1)
                    self.sim_thread.join() # wait for the thread to finish.
                    print("thread has joined")

                # Reset the topology
                self.topology.reset()

                # Wait for network to stabilize
                while self.topology.is_resetting:
                    time.sleep(1)
                print("Network reset complete")


                # start the thread again. 
                self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
                self.sim_thread.start()

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

        # Start Flask server in a separate thread
        self.flask_port = 9000  # Different from global controller's port
        # self.flask_thread = hub.spawn(self.run_flask)

        #  # Start simulation in a separate thread
        # self.sim_thread = hub.spawn(self.run_simulation)

        # Start Flask server in a separate thread
        self.flask_thread = threading.Thread(target=self.run_flask, daemon=True)
        self.flask_thread.start()

        # Start simulation in a separate thread
        self.sim_thread = threading.Thread(target=self.run_simulation, daemon=True)
        self.sim_thread.start()
    
    def run_flask(self):
        """
        Run the Flask server.
        """
        print(f"Starting flask server on port: {self.flask_port}")
        # self.app.run(host="0.0.0.0", port=self.flask_port)
        self.app.run(host="127.0.0.1", port=self.flask_port)
    
    def run_simulation(self):
        # start simulation
        print("Starting simulation")
        print(f"Time scaling factor: {self.topology.time_scale} x (1 day in {24*60/self.topology.time_scale:.1f} minutes)")
        self.topology.run_simulation()

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--total_hours", type=float, default=24.0)
    parser.add_argument("--global_controller_ip", type=str, default="192.168.2.33")
    parser.add_argument("--global_controller_port", type=int, default=8000)
    parser.add_argument("--flow_duration", type=int, default=10)
    parser.add_argument("--time_scale", type=float, default=60.0)
    args = parser.parse_args()

    # Create and start the wrapper
    wrapper = JanosUSTopologyWrapper(args)

    # Keep the main thread running
    try:
        while True:  # Keep the main thread alive
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down topology wrapper...")
