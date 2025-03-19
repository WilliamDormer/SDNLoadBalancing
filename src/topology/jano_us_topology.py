#!/usr/bin/env python3

import sys
import argparse

sys.path.append("/root/mininet/")

import time
import random
import numpy as np
import os
from mininet.net import Mininet
from mininet.node import RemoteController
from mininet.link import TCLink
from mininet.log import info, setLogLevel
from mininet.clean import Cleanup
import threading

# At the beginning of your main code
setLogLevel("info")  # Options: 'debug', 'info', 'warning', 'error', 'critical'


class JanosUSTopology:
    """
    A class that handles the Janos-US network topology creation and management
    for use with a Gym environment.
    """

    def __init__(self, args):
        """
        Initialize the JanosUSTopology.

        Args:
            host_ip (str): IP address of the controller host
            base_lambda (float): Base lambda for the Poisson process
            amplitude (float): Amplitude for the sinusoidal fluctuation
            period (float): Period of the sinusoidal fluctuation
            duration (int): Duration of the simulation
            max_bw (str): Maximum bandwidth for the iperf flows
            flow_duration (int): Duration of each iperf flow
        """
        self.args = args
        self.HOST_IP = args.host_ip
        self.net = None
        self.switches = {}
        self.hosts = {}
        self.controllers = {}

        # Define domains
        self.domain1 = ["s1", "s2", "s3", "s4", "s5", "s6", "s7"]  # Green domain
        self.domain2 = ["s8", "s9", "s10", "s11", "s12", "s13"]  # Yellow domain
        self.domain3 = ["s14", "s15", "s16", "s17", "s18", "s19"]  # Red domain
        self.domain4 = [
            "s20",
            "s21",
            "s22",
            "s23",
            "s24",
            "s25",
            "s26",
        ]  # Turquoise domain

        # Define link structure
        self.links = [
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

        # Define parameters for the simulation
        self.base_lambda = args.base_lambda
        self.amplitude = args.amplitude
        self.period = args.period
        self.max_bw = args.max_bw
        self.flow_duration = args.flow_duration
        self.is_resetting = False
        self.create_network()
        self.start_network()

    def __del__(self):
        """
        Clean up the Mininet network when the object is deleted.
        """
        if self.net:
            self.net.stop()
        os.system("sudo mn -c")
    def create_network(self):
        """
        Create a fresh Mininet network with the Janos-US topology.

        Returns:
            Mininet: The created network object
        """
        # Create a network with remote controllers
        self.net = Mininet(controller=RemoteController, link=TCLink)

        # Add controllers
        info("*** Adding controllers\n")
        self.controllers["g0"] = self.net.addController(
            "g0", controller=RemoteController, ip=self.HOST_IP, port=6653
        )
        self.controllers["c1"] = self.net.addController(
            "c1", controller=RemoteController, ip=self.HOST_IP, port=6654
        )
        self.controllers["c2"] = self.net.addController(
            "c2", controller=RemoteController, ip=self.HOST_IP, port=6655
        )
        self.controllers["c3"] = self.net.addController(
            "c3", controller=RemoteController, ip=self.HOST_IP, port=6656
        )
        self.controllers["c4"] = self.net.addController(
            "c4", controller=RemoteController, ip=self.HOST_IP, port=6657
        )

        # Add switches
        info("*** Adding switches\n")
        self.switches = {}
        for i in range(1, 27):
            self.switches["s{}".format(i)] = self.net.addSwitch(
                "s{}".format(i), stp=True, failMode="standalone"
            )

        # Add hosts
        info("*** Adding hosts\n")
        self.hosts = {}
        for i in range(1, 27):
            self.hosts["h{}".format(i)] = self.net.addHost("h{}".format(i))

        # Create links between hosts and switches
        info("*** Creating host-switch links\n")
        for h, s in zip(self.hosts.values(), self.switches.values()):
            self.net.addLink(h, s)

        # Create links between switches
        info("*** Creating switch-switch links\n")
        for v1, v2 in self.links:
            self.net.addLink(self.switches[v1], self.switches[v2])

        return self.net

    def start_network(self):
        """
        Start the network and assign controllers to their respective switches.
        """
        if not self.net:
            raise RuntimeError("Network not created. Call create_network() first.")

        # Start network
        info("*** Starting network\n")
        self.net.build()

        # Start controllers
        for controller in self.controllers.values():
            controller.start()

        # Start switches with their respective domain controllers
        for switch_name, switch in self.switches.items():
            if switch_name in self.domain1:
                switch.start([self.controllers["c1"], self.controllers["g0"]])
            elif switch_name in self.domain2:
                switch.start([self.controllers["c2"], self.controllers["g0"]])
            elif switch_name in self.domain3:
                switch.start([self.controllers["c3"], self.controllers["g0"]])
            elif switch_name in self.domain4:
                switch.start([self.controllers["c4"], self.controllers["g0"]])

        # Wait for network to stabilize
        info("*** Waiting for network to stabilize\n")
        time.sleep(10)

    def reset(self):
        """
        Reset the network by stopping any existing one and creating a fresh topology.

        Returns:
            Mininet: The newly created network
        """
        # Clean up existing network if it exists
        if self.net:
            info("*** Stopping any existing network\n")
            self.is_resetting = True
            # reset flow tables
            for switch in self.net.switches:
                switch.cmd('ovs-ofctl del-flows {} "priority=1"'.format(switch.name))
            # Kill any running iperf processes
            for host in self.net.hosts:
                host.cmd("killall iperf")
            self.net.stop()

            # Clear references
            self.net = None
            self.switches = {}
            self.hosts = {}
            self.controllers = {}

        # Create fresh network
        self.create_network()

        # Start network
        self.start_network()
        self.is_resetting = False
        return self.net

    def run_simulation(self):
        """Run the traffic simulation with the specified parameters."""
        info("*** Starting traffic simulation\n")
        info(
            f"*** Parameters: base_lambda={self.base_lambda}, amplitude={self.amplitude}, period={self.period}s, duration={self.duration}s\n"
        )
        info("*** Waiting for 10 seconds before starting simulation\n")
        time.sleep(10)

        # Calculate end time
        end_time = time.time() + self.duration

        # Start flow generation
        self.generate_flows(end_time)

        # Start flow table clearing
        self.clear_flow_tables(end_time)

        info(f"*** Simulation will run for {self.duration} seconds\n")

    def start_iperf_flow(self, src_host, dst_host):
        """
        Start an iperf flow between source and destination hosts.

        Args:
            src_host: Source host
            dst_host: Destination host
            max_bw (str): Maximum bandwidth (e.g. "10M")
            duration (int): Duration of the flow in seconds
        """
        port = random.randint(5000, 6000)
        # Start iperf server on destination
        dst_host.cmd(
            f"iperf -s -u -p {port} -t {self.flow_duration+5} > /dev/null 2>&1 &"
        )

        # Start iperf client on source with random bandwidth up to max_bw
        bw = (
            random.randint(1, int(self.max_bw[:-1]))
            if self.max_bw[-1] == "M"
            else random.randint(1, 10)
        )
        bw_str = f"{bw}M"
        src_host.cmd(
            f"iperf -c {dst_host.IP()} -u -p {port} -t {self.flow_duration} -b {bw_str} > /dev/null 2>&1 &"
        )

        info(
            f"  Flow: {src_host.name} -> {dst_host.name} ({bw_str}, {self.flow_duration}s)\n"
        )
        return {
            "src": src_host.name,
            "dst": dst_host.name,
            "bw": bw_str,
            "duration": self.flow_duration,
            "start_time": time.time(),
        }

    def generate_flows(self, end_time):
        """Generate flows based on Poisson process with periodic fluctuation."""
        hosts = self.net.hosts
        current_time = time.time()

        if current_time >= end_time:
            return

        # Calculate current lambda with periodic fluctuation
        t = (current_time % self.period) / self.period
        current_lambda = max(
            0.1, self.base_lambda + self.amplitude * np.sin(2 * np.pi * t)
        )

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
            self.start_iperf_flow(src_host, dst_host)


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_ip", type=str, default="192.168.2.33")
    parser.add_argument("--base_lambda", type=float, default=5.0)
    parser.add_argument("--amplitude", type=float, default=3.0)
    parser.add_argument("--period", type=float, default=60.0)
    parser.add_argument("--max_bw", type=str, default="10M")
    parser.add_argument("--flow_duration", type=int, default=5)
    args = parser.parse_args()

    topology = JanosUSTopology(args)
