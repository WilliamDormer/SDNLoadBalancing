# SDN Load Balancing

This project implements a Software-Defined Networking (SDN) load balancing solution using Ryu controllers and Mininet for network simulation.

## Project Overview

The project simulates a multi-domain SDN network based on the Janos-US topology with:

- 1 global controller
- 4 domain controllers
- 26 switches organized into 4 domains
- 26 hosts (one connected to each switch)

## Prerequisites

- Docker (for containerized execution)
- XQuartz (for macOS users)
- X11 server (for Linux users)

## Getting Started

### Option 1: Running with Docker (Recommended)

1. Clone this repository:

   ```
   git clone <repository-url>
   cd SDNLoadBalancing
   ```

2. Run the project using the provided script:

   ```
   bash scripts/run.sh
   ```

   This script will:

   - Check for necessary dependencies (Docker, XQuartz on macOS, X11 on Linux)
   - Build the Docker image if it doesn't exist
   - Create and start a container with all required networking configurations
   - Attach to the container's shell

3. Outside of the docker, start the controllers:

   ```
   bash scripts/run_controllers.sh
   ```

   This will start:

   - One global controller (g0) on port 6653
   - Four domain controllers (c1, c2, c3, c4) on ports 6654-6657

4. In the docker container, run the Janos-US topology simulation:

   ```
   python src/janos_us_topology.py
   ```

   This will create the network topology with switches, hosts, and links, and connect them to the appropriate controllers.

## Project Structure

- `src/`: Source code for the controllers and topology
  - `ryu_controller.py`: Implementation of the SDN controllers
  - `janos_us_topology.py`: Mininet topology definition
- `scripts/`: Helper scripts
  - `run.sh`: Script to set up and run the Docker container
  - `run_controllers.sh`: Script to start all controllers
- `logs/`: Log files for the controllers

## Troubleshooting

### X11 Forwarding Issues

- **macOS**: Make sure XQuartz is installed and running. You may need to restart XQuartz and run `xhost + localhost` in a terminal.
- **Linux**: Run `xhost +local:` to allow local connections to the X server.

### Controller Connection Issues

If switches cannot connect to controllers, check:

1. The HOST_IP variable in `src/janos_us_topology.py` matches your machine's IP address
2. All required ports (6653-6657) are open and not used by other applications
