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
import math
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
            rate (float): Rate of the Poisson process
            amplitude (float): Amplitude for the sinusoidal fluctuation
            period (float): Period of the sinusoidal fluctuation
            duration (int): Duration of the simulation
            max_bw (str): Maximum bandwidth for the iperf flows
            flow_duration (int): Duration of each iperf flow
            time_scale (float): Time scale for the simulation
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
        self.base_rate = args.base_rate
        self.fluctuation_amplitude = args.fluctuation_amplitude
        self.period_hours = args.period_hours
        self.total_hours = args.total_hours
        self.max_bw = args.max_bw
        self.flow_duration = args.flow_duration
        self.time_scale = args.time_scale
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
    
    def generate_poisson_with_fluctuation(self):
        """
        Generate event times following a Poisson process with periodic fluctuation.
        
        Args:
            base_rate: Base rate of the Poisson process (events per hour)
            fluctuation_amplitude: Amplitude of the sinusoidal fluctuation (0-1 scale)
            period_hours: Period of fluctuation in hours
            total_hours: Total simulation time in hours
        
        Returns:
            List of event times in seconds
        """
        total_seconds = self.total_hours * 3600
        time_points = []
        current_time = 0
        
        while current_time < total_seconds:
            # Calculate time-varying rate using sinusoidal fluctuation
            hour_of_day = (current_time / 3600) % self.period_hours
            fluctuation_factor = 1 + self.fluctuation_amplitude * math.sin(2 * math.pi * hour_of_day / self.period_hours)
            current_rate = self.base_rate * fluctuation_factor / 3600  # Convert to per-second rate
            
            # Generate next interval using current rate
            next_interval = np.random.exponential(1 / current_rate)
            current_time += next_interval
            
            if current_time < total_seconds:
                time_points.append(current_time)
        
        return np.array(time_points)



    def run_simulation(self):
        """Run the traffic simulation with the specified parameters."""
        info("*** Starting traffic simulation\n")
        info(
            f"*** Parameters: base_rate={self.base_rate}, fluctuation_amplitude={self.fluctuation_amplitude}, period_hours={self.period_hours}, total_hours={self.total_hours}\n"
        )
        info("*** Waiting for 10 seconds before starting simulation\n")
        time.sleep(10)
        hosts = self.net.hosts
        num_hosts = len(hosts)

        # Generate event times
        info(f"Generating Poisson events with periodical fluctuation...\n")
        event_times = self.generate_poisson_with_fluctuation()
        
        info(f"Scheduled {len(event_times)} flow requests over {self.total_hours} hours\n")

        # start the simulation
        start_time = time.time()

        for i, event_time in enumerate(event_times):
            # Scale time for testing
            scaled_time = event_time / self.time_scale
            
            # Wait until it's time for this event
            time_to_wait = scaled_time - (time.time() - start_time)
            if time_to_wait > 0:
                time.sleep(time_to_wait)
                
            # Calculate current hour for logging
            current_hour = (event_time / 3600) % self.period_hours
            
            # Generate random number of flows for this event
            num_flows = np.random.poisson(self.base_rate)
            num_flows = min(num_flows,10)

            for _ in range(num_flows):
                # Select random source and destination hosts
                src_idx = np.random.randint(0, num_hosts)
                dst_idx = np.random.randint(0, num_hosts)
                while dst_idx == src_idx:
                    dst_idx = np.random.randint(0, num_hosts)
                
                src_host = hosts[src_idx]
                dst_host = hosts[dst_idx]
                
                self.start_iperf_flow(src_host, dst_host, current_hour,i)
        
            # Kill any stalled iperf processes periodically
            if i % 10 == 0:
                for host in hosts:
                    host.cmd("pkill -9 -f 'iperf -s' > /dev/null 2>&1")

    def start_iperf_flow(self, src_host, dst_host, current_hour, idx):
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
            f"  Flow {idx+1} at {current_hour:.2f}h: {src_host.name} -> {dst_host.name} ({bw_str}, {self.flow_duration}s)\n"
        )
        return {
            "src": src_host.name,
            "dst": dst_host.name,
            "bw": bw_str,
            "duration": self.flow_duration,
            "start_time": time.time(),
        }


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--host_ip", type=str, default="192.168.2.33")
    parser.add_argument("--base_rate", type=float, default=5.0)
    parser.add_argument("--fluctuation_amplitude", type=float, default=0.5)
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--total_hours", type=float, default=24.0)
    parser.add_argument("--max_bw", type=str, default="10M")
    parser.add_argument("--flow_duration", type=int, default=10)
    parser.add_argument("--time_scale", type=float, default=60.0)
    args = parser.parse_args()

    topology = JanosUSTopology(args)
