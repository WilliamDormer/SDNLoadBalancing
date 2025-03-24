#!/usr/bin/env python3

from topology.jano_us_topology import JanosUSTopology
from flask import Flask, jsonify, request
import argparse
import numpy as np
import time
import threading

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
        self.flask_port = 9000

        # Initialize topology
        self.topology = JanosUSTopology(args)
        self.reset_complete = False
        self.reset_error = None

        # Track current state
        self.current_state = np.zeros((self.num_domains, self.num_switches))
        self.start_time = time.time()

        # Register routes
        @self.app.route("/reset_topology", methods=["POST"])
        def reset_topology():
            """Handle reset requests synchronously"""
            print("Received reset request")
            try:
                # Reset the flags
                self.reset_complete = False
                self.reset_error = None

                # Start the reset in a separate thread
                thread = threading.Thread(target=self._do_reset)
                thread.start()

                # Wait for reset with timeout
                start_time = time.time()
                timeout = 30  # seconds

                while not self.reset_complete and time.time() - start_time < timeout:
                    time.sleep(0.1)
                    if self.reset_error:
                        raise Exception(self.reset_error)

                if not self.reset_complete:
                    raise Exception("Reset operation timed out")

                print("Network reset complete")
                thread.join()
                return jsonify({"message": "Topology reset successful"}), 200
            except Exception as e:
                print(f"Reset failed with error: {str(e)}")
                return jsonify({"error": f"Reset failed: {str(e)}"}), 500

        # Start Flask server in a separate thread
        self.flask_port = 9000  # Different from global controller's port
        # self.flask_thread = hub.spawn(self.run_flask)

        print("Waiting for the reset to be called before starting simulation")
        print(
            f"Time scaling factor: {self.topology.time_scale} x (1 day in {24*60/self.topology.time_scale:.1f} minutes)"
        )

    def _do_reset(self):
        """Execute reset operation in a separate thread"""
        try:
            # Perform reset
            print("Resetting topology")
            self.topology.reset()

            # Wait for reset to complete
            while self.topology.is_resetting.value:
                time.sleep(0.1)
            # Mark reset as complete
            self.reset_complete = True
        except Exception as e:
            self.reset_error = str(e)
            self.reset_complete = True  # Mark as complete even on error

    def run_flask(self):
        """
        Run the Flask server.
        """
        print(f"Starting Flask server on port {self.flask_port}...")
        self.app.run(host="0.0.0.0", port=self.flask_port, debug=False)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--total_hours", type=float, default=24.0)
    parser.add_argument("--global_controller_ip", type=str, default="192.168.2.33")
    parser.add_argument("--global_controller_port", type=int, default=8000)
    parser.add_argument("--flow_duration", type=int, default=2)
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
