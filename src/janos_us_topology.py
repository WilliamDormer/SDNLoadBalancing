#!/usr/bin/env python3
# import the mininet library
import sys

sys.path.append("/root/mininet/")

from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import numpy as np
import threading
import time
import random
import argparse

HOST_IP = "192.168.2.33"


def parse_args():
    """Parse command line arguments for the simulation."""
    parser = argparse.ArgumentParser(
        description="Run Janos-US topology with Poisson traffic generation"
    )
    parser.add_argument(
        "--base-lambda",
        type=float,
        default=5.0,
        help="Base rate parameter for Poisson process",
    )
    parser.add_argument(
        "--amplitude", type=float, default=3.0, help="Amplitude of periodic fluctuation"
    )
    parser.add_argument(
        "--period", type=float, default=60.0, help="Period of fluctuation in seconds"
    )
    parser.add_argument(
        "--duration", type=int, default=300, help="Duration of simulation in seconds"
    )
    parser.add_argument(
        "--max-bw", type=str, default="10M", help="Maximum bandwidth for iperf flows"
    )
    parser.add_argument(
        "--flow-duration",
        type=int,
        default=5,
        help="Duration of each iperf flow in seconds",
    )
    return parser.parse_args()


def start_iperf_flow(src_host, dst_host, max_bw="10M", duration=5):
    """Start an iperf flow between source and destination hosts."""
    port = random.randint(5000, 6000)
    # Start iperf server on destination
    dst_host.cmd(f"iperf -s -u -p {port} -t {duration+5} > /dev/null 2>&1 &")
    # Start iperf client on source with random bandwidth up to max_bw
    bw = (
        random.randint(1, int(max_bw[:-1]))
        if max_bw[-1] == "M"
        else random.randint(1, 10)
    )
    bw_str = f"{bw}M"
    src_host.cmd(
        f"iperf -c {dst_host.IP()} -u -p {port} -t {duration} -b {bw_str} > /dev/null 2>&1 &"
    )
    info(f"  Flow: {src_host.name} -> {dst_host.name} ({bw_str}, {duration}s)\n")


def clear_flow_tables(net, end_time, interval=30):
    """Periodically clear flow tables to force new packet-in messages."""
    current_time = time.time()

    if current_time >= end_time:
        return

    info(f"*** Clearing flow tables at {time.strftime('%H:%M:%S')}\n")
    for switch in net.switches:
        switch.cmd('ovs-ofctl del-flows {} "priority=1"'.format(switch.name))

    # Schedule next clearing
    threading.Timer(interval, clear_flow_tables, [net, end_time, interval]).start()


def generate_flows(
    net, base_lambda, amplitude, period, end_time, max_bw, flow_duration
):
    """Generate flows based on Poisson process with periodic fluctuation."""
    hosts = net.hosts
    current_time = time.time()

    if current_time >= end_time:
        return

    # Calculate current lambda with periodic fluctuation
    t = (current_time % period) / period
    current_lambda = max(0.1, base_lambda + amplitude * np.sin(2 * np.pi * t))

    # Generate number of flows using Poisson distribution
    num_flows = np.random.poisson(current_lambda)
    num_flows = min(num_flows, 10)  # Limit max concurrent flows to avoid overload

    # Log current lambda and number of flows
    info(
        f"Time: {time.strftime('%H:%M:%S')}, Lambda: {current_lambda:.2f}, Flows: {num_flows}\n"
    )

    # Generate each flow
    for _ in range(num_flows):
        src_host = random.choice(hosts)
        dst_host = random.choice([h for h in hosts if h != src_host])

        # Start iperf traffic
        start_iperf_flow(src_host, dst_host, max_bw, flow_duration)

    # Schedule next flow generation (in 1 second)
    threading.Timer(
        1.0,
        generate_flows,
        [net, base_lambda, amplitude, period, end_time, max_bw, flow_duration],
    ).start()


def run_simulation(net, args):
    """Run the traffic simulation with the specified parameters."""
    info("*** Starting traffic simulation\n")
    info(
        f"*** Parameters: base_lambda={args.base_lambda}, amplitude={args.amplitude}, period={args.period}s, duration={args.duration}s\n"
    )
    info("*** Waiting for 10 seconds before starting simulation\n")
    time.sleep(10)

    # Calculate end time
    end_time = time.time() + args.duration

    # Start flow generation
    generate_flows(
        net,
        args.base_lambda,
        args.amplitude,
        args.period,
        end_time,
        args.max_bw,
        args.flow_duration,
    )

    # Start flow table clearing
    clear_flow_tables(net, end_time)

    info(f"*** Simulation will run for {args.duration} seconds\n")


def create_topology(args=None):
    # Create a network with remote controllers
    net = Mininet(controller=RemoteController, link=TCLink)

    # Add controllers (1 global + 4 domain controllers)
    info("*** Adding controllers\n")
    # Global controller
    g0 = net.addController("g0", controller=RemoteController, ip=HOST_IP, port=6653)

    # Domain controllers
    c1 = net.addController("c1", controller=RemoteController, ip=HOST_IP, port=6654)
    c2 = net.addController("c2", controller=RemoteController, ip=HOST_IP, port=6655)
    c3 = net.addController("c3", controller=RemoteController, ip=HOST_IP, port=6656)
    c4 = net.addController("c4", controller=RemoteController, ip=HOST_IP, port=6657)

    # Add switches (representing nodes v1-v26 in Janos-US topology)
    info("*** Adding switches\n")
    switches = {}
    for i in range(1, 27):
        switches["s{}".format(i)] = net.addSwitch(
            "s{}".format(i), stp=True, failMode="standalone"
        )

    # Add hosts (representing nodes h1-h26 in Janos-US topology)
    info("*** Adding hosts\n")
    hosts = {}
    for i in range(1, 27):
        hosts["h{}".format(i)] = net.addHost("h{}".format(i))

    # Define domains based on the image
    domain1 = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]  # Green domain
    domain2 = ["s8", "s9", "s10", "s11", "s12", "s13"]  # Yellow domain
    domain3 = ["s14", "s15", "s16", "s17", "s18", "s19"]  # Red domain
    domain4 = [
        "s20",
        "s21",
        "s22",
        "s23",
        "s24",
        "s25",
        "s26",
    ]  # Turquoise domain

    # Add links based on Janos-US topology from the image
    info("*** Creating links\n")
    links = [
        # Domain 1 internal links
        ("s1", "s2"),
        ("s1", "s5"),
        ("s2", "s3"),
        ("s2", "s5"),
        ("s3", "s6"),
        ("s3", "s4"),
        ("s4", "s5"),
        ("s4", "s6"),
        ("s5", "s7"),
        # Domain 2 internal links
        ("s8", "s9"),
        ("s8", "s11"),
        ("s9", "s10"),
        ("s9", "s11"),
        ("s10", "s12"),
        ("s11", "s12"),
        ("s11", "s13"),
        ("s12", "s13"),
        # Domain 3 internal links
        ("s14", "s15"),
        ("s15", "s16"),
        ("s15", "s19"),
        ("s16", "s17"),
        ("s16", "s18"),
        ("s17", "s18"),
        ("s18", "s19"),
        # Domain 4 internal links
        ("s20", "s21"),
        ("s20", "s23"),
        ("s21", "s22"),
        ("s21", "s25"),
        ("s22", "s25"),
        ("s23", "s24"),
        ("s24", "s25"),
        ("s24", "s26"),
        ("s25", "s26"),
        # Inter-domain links
        ("s6", "s20"),
        ("s6", "s23"),
        ("s7", "s9"),
        ("s7", "s20"),
        ("s8", "s20"),
        ("s12", "s14"),
        ("s13", "s15"),
        ("s13", "s21"),
        ("s19", "s22"),
    ]

    # Create links between hosts and switches
    for h, s in zip(hosts.values(), switches.values()):
        net.addLink(h, s)

    # Create links between switches
    for v1, v2 in links:
        net.addLink(switches[v1], switches[v2])
    # Start network
    info("*** Starting network\n")
    net.build()

    # Start controllers
    g0.start()
    c1.start()
    c2.start()
    c3.start()
    c4.start()

    # Start switches with their respective domain controllers
    for switch_name, switch in switches.items():
        if switch_name in domain1:
            switch.start([c1, g0])
        elif switch_name in domain2:
            switch.start([c2, g0])
        elif switch_name in domain3:
            switch.start([c3, g0])
        elif switch_name in domain4:
            switch.start([c4, g0])

    # Start the traffic simulation if args are provided
    if args:
        # Start simulation in a separate thread to not block CLI
        sim_thread = threading.Thread(target=run_simulation, args=(net, args))
        sim_thread.daemon = True
        sim_thread.start()

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    # Ensure all iperf processes are killed
    for host in net.hosts:
        host.cmd("killall iperf")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    args = parse_args()
    create_topology(args)
