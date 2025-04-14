#!/bin/bash

# This script runs 5 instances of the custom ryu_controller.py:
# - One global controller (g0) on port 6653
# - Four domain controllers (c1, c2, c3, c4) on ports 6654-6657

# Kill any existing Ryu controllers
echo "Stopping any existing Ryu controllers..."
pkill -f ryu-manager || true
sleep 2

# Get the project directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$( dirname "${SCRIPT_DIR}" )" && pwd )"
echo "Project directory: $PROJECT_DIR"

# # Activate the virtual environment
# source venv/bin/activate

CONFIG_FILE="src/ryu_config_files/janos_us.conf"

# Function to run a controller in the background
run_controller() {
    local name=$1
    local port=$2
    echo "Starting controller $name on port $port..."
    ryu-manager --ofp-tcp-listen-port=$port --config-file="$CONFIG_FILE" "$PROJECT_DIR/src/ryu_controller.py" > "logs/$name.log" 2>&1 &
    sleep 1  # Give it a moment to start
    echo "Controller $name started with PID $!"
}

run_global_controller() {
    echo "Starting global controller on port 6653..."
    ryu-manager --ofp-tcp-listen-port=6653 --config-file="$CONFIG_FILE" "$PROJECT_DIR/src/ryu_global_controller.py" > "logs/g0.log" 2>&1 &
    sleep 1  # Give it a moment to start
    echo "Global controller started with PID $!"
}

# Start the global controller (g0)
run_global_controller

# Start the domain controllers (c1-c4)
run_controller "c1" 6654
run_controller "c2" 6655
run_controller "c3" 6656
run_controller "c4" 6657

echo "All controllers started. Logs are being written to g0.log, c1.log, c2.log, c3.log, and c4.log"
echo "To stop all controllers, run: pkill -f ryu-manager"

# Wait for all controllers to be ready
echo "Waiting for controllers to initialize..."
sleep 5

echo "Controllers are ready. You can now run the Janos US topology"