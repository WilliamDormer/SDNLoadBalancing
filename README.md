# SDN Load Balancing

This project implements a Software-Defined Networking (SDN) load balancing solution using Ryu controllers and Mininet for network simulation.

## Project Overview

The project simulates a multi-domain SDN network based on the Janos-US topology with:

- 1 global controller
- 4 domain controllers
- 26 switches organized into 4 domains
- 26 hosts (one connected to each switch)

The simulation includes traffic generation capabilities that simulate packet-in messages with different rates following a Poisson process with periodic fluctuation.

## Prerequisites

- Docker (for containerized execution)
- XQuartz (for macOS users)
- X11 server (for Linux users)
- Python packages:
  - numpy (for Poisson process simulation)
  - mininet
  - ryu

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

## Traffic Simulation

The `janos_us_topology.py` script now includes traffic simulation capabilities that generate packet-in messages with varying rates. The traffic generation follows a Poisson process with a periodic fluctuation component.

### How It Works

1. Flow requests are generated according to a Poisson process
2. The rate parameter (lambda) of the Poisson process fluctuates periodically using a sine wave
3. For each flow, random source and destination hosts are selected
4. Iperf is used to generate UDP traffic between the hosts
5. Flow tables are periodically cleared to force new packet-in messages

### Command Line Arguments

The simulation can be customized with the following command line arguments:

- `--base-lambda`: Base rate parameter for the Poisson process (default: 5.0)
- `--amplitude`: Amplitude of the periodic fluctuation (default: 3.0)
- `--period`: Period of the fluctuation in seconds (default: 60.0)
- `--duration`: Duration of the simulation in seconds (default: 300)
- `--max-bw`: Maximum bandwidth for iperf flows (default: 10M)
- `--flow-duration`: Duration of each iperf flow in seconds (default: 5)

### Example Usage

Run with default parameters:

```
python src/janos_us_topology.py
```

Run with custom parameters:

```
python src/janos_us_topology.py --base-lambda 3.0 --amplitude 2.0 --period 30.0 --duration 600 --max-bw 5M --flow-duration 3
```

This will:

- Set the base lambda to 3.0 flows per second
- Set the amplitude of fluctuation to 2.0
- Set the period of fluctuation to 30 seconds
- Run the simulation for 10 minutes (600 seconds)
- Limit the maximum bandwidth of each flow to 5 Mbps
- Set each flow to last 3 seconds

## Project Structure

- `src/`: Source code for the controllers and topology
  - `ryu_controller.py`: Implementation of the SDN controllers
  - `janos_us_topology.py`: Mininet topology definition with traffic simulation
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

### Traffic Simulation Issues

If you encounter issues with the traffic simulation:

1. Ensure numpy is installed (`pip install numpy`)
2. Try reducing the base-lambda value if the system becomes overloaded
3. Check that iperf is installed in the container
4. If iperf processes don't terminate properly, manually run `killall iperf` in the container
