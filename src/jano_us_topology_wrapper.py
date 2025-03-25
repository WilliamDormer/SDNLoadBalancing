#!/usr/bin/env python3

from topology.jano_us_topology import JanosUSTopology
from flask import Flask, jsonify, request
import argparse
import numpy as np
import time
import threading
import subprocess
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


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
            logger.info("Received reset request")
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

                logger.info("Network reset complete")
                thread.join()
                return jsonify({"message": "Topology reset successful"}), 200
            except Exception as e:
                logger.error(f"Reset failed with error: {str(e)}")
                return jsonify({"error": f"Reset failed: {str(e)}"}), 500

        @self.app.route("/migrate_switch", methods=["POST"])
        def migrate_switch():
            """Handle switch migration requests"""
            try:
                data = request.get_json()

                # Log the received request
                logger.info("=" * 50)
                logger.info("Migration Request Received:")
                logger.info(f"Raw data: {data}")

                if not data or "target_controller" not in data or "switch" not in data:
                    logger.error("Missing required fields in request")
                    return jsonify({"error": "Missing required fields"}), 400

                switch_id = data["switch"]
                target_controller = data["target_controller"]

                logger.info(f"Attempting migration:")
                logger.info(f"Switch ID: {switch_id}")
                logger.info(f"Target Controller: {target_controller}")

                # Get the controller object
                controller = self.topology.controllers[f"c{target_controller}"]

                # Access controller properties directly
                controller_ip = controller.ip
                controller_port = controller.port

                logger.info(f"Controller Details:")
                logger.info(f"IP: {controller_ip}")
                logger.info(f"Port: {controller_port}")

                # Execute the migration command
                cmd = f"ovs-vsctl set-controller s{switch_id} tcp:{controller_ip}:{controller_port}"
                logger.info(f"Executing command: {cmd}")

                result = subprocess.run(cmd, shell=True)
                if result.returncode != 0:
                    logger.error(f"Migration command failed:")
                    logger.error(f"Error output: {result.stderr}")
                    raise Exception(f"Migration command failed: {result.stderr}")

                logger.info("Migration command executed successfully")
                logger.info("=" * 50)
                return jsonify({"message": "Migration successful"}), 200

            except Exception as e:
                logger.error(f"Migration failed with error: {str(e)}")
                logger.info("=" * 50)
                return jsonify({"error": f"Migration failed: {str(e)}"}), 500

        # Start Flask server in a separate thread
        self.flask_port = 9000  # Different from global controller's port

        print("Waiting for the reset to be called before starting simulation")
        print(
            f"Time scaling factor: {self.topology.time_scale} x (1 day in {24*60/self.topology.time_scale:.1f} minutes)"
        )
        self.run_flask()

    def _do_reset(self):
        """Execute reset operation in a separate thread"""
        try:
            # Perform reset
            logger.info("Resetting topology")
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
        # logger.info(f"Starting Flask server on port {self.flask_port}...")
        self.app.run(host="0.0.0.0", port=self.flask_port, debug=False)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--total_hours", type=float, default=24.0)
    parser.add_argument("--global_controller_ip", type=str, default="192.168.2.33")
    parser.add_argument("--global_controller_port", type=int, default=8000)
    parser.add_argument("--flow_duration", type=int, default=2)
    parser.add_argument("--time_scale", type=float, default=30.0)
    args = parser.parse_args()

    # Log startup information
    logger.info("Starting JanosUSTopologyWrapper")
    wrapper = JanosUSTopologyWrapper(args)

    # Keep the main thread running
    try:
        while True:  # Keep the main thread alive
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down topology wrapper...")
