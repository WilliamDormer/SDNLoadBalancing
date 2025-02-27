# SDNLoadBalancing
A final project for UofT Computer Networks for Machine Learning graduate course

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Structure](#project-structure)
3. [Setup Instructions](#setup-instructions)
4. [Network Testing](#network-testing)
5. [Controller Functionality](#controller-functionality)
## Prerequisites

Before running the project, ensure you have the following installed:

- Docker with the mininet-in-a-container image
- Python 3.6 or higher
- Ryu SDN Framework
- XQuartz (for macOS) or X11 server (for Linux) if using MiniEdit

## Project Structure

The project contains the following key files:

| File                | Description                                  |
| ------------------- | -------------------------------------------- |
| `ryu_controller.py` | Ryu controller implementing L2 switch        |
| `simple_topo.py`    | Mininet topology with 2 switches and 4 hosts |

## Setup Instructions

1. Start the Mininet container:

   ```bash
   ./runmininet.sh
   ```

2. Inside the container, start the Ryu controller:

   ```bash
   cd /root/SDNLoadBalancing
   ryu-manager ryu_controller.py
   ```

3. In another terminal, connect to the running container and start the topology:
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

## Controller Functionality

The L2Switch controller in `ryu_controller.py` implements:

1. Basic packet flooding for all incoming packets
2. OFPPacketOut message handling
3. Buffer management for packet data
4. OpenFlow 1.3 protocol support
