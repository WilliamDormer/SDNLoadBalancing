# SDNLoadBalancing

A final project for UofT Computer Networks for Machine Learning graduate course

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
   - [Using Docker (Recommended)](#using-docker-recommended)
     - [macOS Setup](#macos-setup)
   - [Manual Setup](#manual-setup)
4. [Network Testing](#network-testing)
5. [Controller Functionality](#controller-functionality)

## Prerequisites

Before running the project, ensure you have the following installed:

- Docker and Docker Compose (for containerized setup)
- X11 server:
  - For macOS: XQuartz

For manual setup:

- Python 3.6 or higher
- Mininet
- Ryu SDN Framework
- Open vSwitch

## Project Structure

The project contains the following key files:

| File                 | Description                                  |
| -------------------- | -------------------------------------------- |
| `ryu_controller.py`  | Ryu controller implementing L2 switch        |
| `simple_topo.py`     | Mininet topology with 2 switches and 4 hosts |
| `Dockerfile`         | Docker configuration for the environment     |
| `docker-compose.yml` | Docker Compose configuration                 |
| `run.sh`             | Script to run the Docker container           |

## Setup Instructions

### Using Docker (Recommended)

The easiest way to run this project is using Docker, which packages all dependencies and configurations.

#### 1. Clone the repository

```bash
git clone https://github.com/yourusername/SDNLoadBalancing.git
cd SDNLoadBalancing
```

#### 2. Run the container

```bash
chmod +x run.sh  # Make sure the script is executable
./run.sh
```

This script will:

- Check for required dependencies
- Configure X11 forwarding
- Build and start the Docker container
- Attach to the container

#### macOS Setup

If you're using macOS:

1. Install Docker Desktop for Mac from [Docker's website](https://www.docker.com/products/docker-desktop)

2. Install XQuartz:

   ```bash
   brew install --cask xquartz
   ```

3. Open XQuartz, go to Preferences > Security, and check "Allow connections from network clients"

4. Restart XQuartz

#### 3. Inside the container, start the Ryu controller

```bash
cd /root/SDNLoadBalancing
ryu-manager ryu_controller.py
```

#### 4. In another terminal, connect to the running container

```bash
docker exec -it sdn-loadbalancer bash
```

#### 5. Start the Mininet topology

```bash
cd /root/SDNLoadBalancing
python3 simple_topo.py
```

## Network Testing

Once both the controller and topology are running, you can test connectivity:

- Test all connections:

  ```bash
  mininet> pingall
  ```

- Test individual connections:

  ```bash
  mininet> h1 ping h3
  ```

- View the flow tables:
  ```bash
  mininet> sh ovs-ofctl dump-flows s1
  ```

## Controller Functionality

The L2Switch controller in `ryu_controller.py` implements:

1. Basic packet flooding for all incoming packets
2. OFPPacketOut message handling
3. Buffer management for packet data
4. OpenFlow 1.3 protocol support

## Troubleshooting

### X11 Forwarding Issues

If you encounter X11 forwarding issues:

- For macOS:
  1. Make sure XQuartz is installed and running
  2. In XQuartz preferences, ensure "Allow connections from network clients" is checked
  3. Restart XQuartz

### Container Networking Issues

If the container cannot connect to the controller:

1. Check that the controller is running
2. Verify that the controller IP and port in `simple_topo.py` are correct
3. Ensure the required ports (6633, 6653) are not blocked by a firewall
