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
import multiprocessing

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
        self.controller_ip = args.global_controller_ip
        self.net = None
        self.switches = {}
        self.hosts = {}
        self.controllers = {}
        self.simulation_process = None  # Replace thread with process
        self.stop_simulation = multiprocessing.Value(
            "b", False
        )  # Shared value for process control
        self.is_resetting = multiprocessing.Value("b", False)
        self.processes = []  # List to keep track of all processes

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

        self.src_dst_pairs = [
            # Extremely heavy traffic in Domain 4 (switches 20-26)
            (20, 11),
            (20, 13),
            (20, 1),  # Domain 4 -> Multiple domains
            (21, 13),
            (21, 2),
            (21, 15),  # Domain 4 -> Multiple domains
            (22, 25),
            (22, 26),
            (22, 24),  # Internal Domain 4 (heavy internal traffic)
            (23, 1),
            (23, 8),
            (23, 14),  # Domain 4 -> Multiple domains
            (24, 5),
            (24, 25),
            (24, 26),  # Domain 4 mix of internal and external
            (25, 17),
            (25, 22),
            (25, 20),  # Domain 4 mix of internal and external
            (26, 22),
            (26, 23),
            (26, 21),  # More internal Domain 4 traffic
            # Medium-heavy traffic in Domain 1 (switches 1-7)
            (1, 6),
            (1, 20),  # Internal and external
            (2, 5),
            (2, 21),  # Internal and external
            (3, 20),
            (3, 4),  # Internal and external
            (4, 21),
            (4, 7),  # Internal and external
            # Light traffic in Domain 2 (switches 8-13)
            (8, 6),  # Domain 2 -> Domain 1
            (9, 19),  # Domain 2 -> Domain 3
            (10, 2),  # Domain 2 -> Domain 1
            # Minimal traffic in Domain 3 (switches 14-19)
            (14, 19),  # Single internal flow
            (15, 16),  # Single internal flow
        ]

        self.base_rates = [
            # Domain 1 (switches 1-7): Medium-high base rate
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            # Domain 2 (switches 8-13): Low base rate
            10,
            10,
            10,
            10,
            10,
            10,
            # Domain 3 (switches 14-19): Very low base rate
            5,
            5,
            5,
            5,
            5,
            5,
            # Domain 4 (switches 20-26): Extremely high base rate
            50,
            50,
            50,
            50,
            50,
            50,
            50,
        ]

        self.fluctuation_amplitudes = [
            # Domain 1: Medium fluctuation
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            0.4,
            # Domain 2: Low fluctuation
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            0.3,
            # Domain 3: Minimal fluctuation
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            0.1,
            # Domain 4: High fluctuation
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
            0.6,
        ]

        # Define parameters for the simulation
        self.period_hours = args.period_hours
        self.total_hours = args.total_hours
        self.flow_duration = args.flow_duration
        self.time_scale = args.time_scale
        self.is_started = False
        # self.create_network()
        # self.start_network()

    def __del__(self):
        """
        Clean up the Mininet network when the object is deleted.
        """
        if self.net:
            self.net.stop()
        os.system("sudo mn -c")

    def stop(self):
        """
        Clean up topology resources
        """
        try:
            # Kill any running iperf processes first
            if hasattr(self, "net") and self.net:
                for host in self.net.hosts:
                    host.cmd("killall -9 iperf")

            # Stop all processes
            if hasattr(self, "processes"):
                for p in self.processes:
                    if p and p.is_alive():
                        p.terminate()
                        p.join()

            # Set stop flags
            if hasattr(self, "stop_simulation"):
                self.stop_simulation.value = True
            if hasattr(self, "is_resetting"):
                self.is_resetting.value = True

            # Stop the network
            if hasattr(self, "net") and self.net:
                info("*** Stopping network\n")
                # Clear flow tables
                for switch in self.net.switches:
                    switch.cmd(
                        'ovs-ofctl del-flows {} "priority=1"'.format(switch.name)
                    )
                self.net.stop()

            # Clean up OVS configurations
            info("*** Cleaning up...\n")
            os.system("sudo mn -c")

            # Clear references
            self.net = None
            self.switches = {}
            self.hosts = {}
            self.controllers = {}
            self.processes = []

        except Exception as e:
            info(f"*** Error during topology cleanup: {str(e)}\n")
            # Continue with cleanup even if there's an error
            try:
                os.system("sudo mn -c")
            except:
                pass
            raise

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
            "g0", controller=RemoteController, ip=self.controller_ip, port=6653
        )
        self.controllers["c1"] = self.net.addController(
            "c1", controller=RemoteController, ip=self.controller_ip, port=6654
        )
        self.controllers["c2"] = self.net.addController(
            "c2", controller=RemoteController, ip=self.controller_ip, port=6655
        )
        self.controllers["c3"] = self.net.addController(
            "c3", controller=RemoteController, ip=self.controller_ip, port=6656
        )
        self.controllers["c4"] = self.net.addController(
            "c4", controller=RemoteController, ip=self.controller_ip, port=6657
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
        """
        # Set flags to stop processes
        self.is_resetting.value = True
        self.stop_simulation.value = True

        if self.net and self.is_started:
            info("*** Stopping any existing network\n")
            # Kill any running iperf processes
            for host in self.net.hosts:
                host.cmd("killall -9 iperf")

            info("*** Terminating all processes\n")
            # Terminate all processes
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
                    p.join()

            info("*** All processes terminated\n")

            # Reset flow tables and stop network
            for switch in self.net.switches:
                switch.cmd('ovs-ofctl del-flows {} "priority=1"'.format(switch.name))
            self.net.stop()
            os.system("sudo mn -c")

            # Clear references
            self.net = None
            self.switches = {}
            self.hosts = {}
            self.controllers = {}

        else:
            info("*** Starting network\n")
            self.is_started = True

        # Create fresh network
        self.create_network()
        self.start_network()

        # Reset flags
        self.is_resetting.value = False
        self.stop_simulation.value = False

        # Prepare simulation processes
        info("*** Preparing traffic simulation\n")
        info(
            f"*** Parameters: period_hours={self.period_hours}, total_hours={self.total_hours}\n"
        )

        # Generate time points
        time_points = {}
        for src, dst in self.src_dst_pairs:
            base_rate = self.base_rates[src - 1]
            fluctuation_amplitude = self.fluctuation_amplitudes[src - 1]
            time_points[(src, dst)] = self.generate_poisson_with_fluctuation(
                base_rate, fluctuation_amplitude
            )
        start_time = time.time()

        # Create processes for each src-dst pair
        self.processes = []
        for src_dst_pair, points in time_points.items():
            process = multiprocessing.Process(
                target=self.run_poisson_process,
                args=(
                    src_dst_pair,
                    points.copy(),
                    self.total_hours * 3600,
                    self.time_scale,
                    start_time,
                    self.stop_simulation,
                    self.is_resetting,
                ),
            )
            process.daemon = True
            self.processes.append(process)

        # Start all processes non-blocking
        for p in self.processes:
            p.start()

        return self.net

    def generate_poisson_with_fluctuation(self, base_rate, fluctuation_amplitude):
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
            fluctuation_factor = 1 + fluctuation_amplitude * math.sin(
                2 * math.pi * hour_of_day / self.period_hours
            )
            current_rate = (
                base_rate * fluctuation_factor / 3600
            )  # Convert to per-second rate

            # Generate next interval using current rate
            next_interval = np.random.exponential(1 / current_rate)
            current_time += next_interval

            if current_time < total_seconds:
                time_points.append(current_time)

        return np.array(time_points)

    def run_poisson_process(
        self,
        src_dst_pair,
        time_points,
        duration,
        time_scale,
        start_time,
        stop_flag,
        reset_flag,
    ):
        """Run the Poisson process for a given src-dst pair."""
        try:
            end_time = start_time + duration
            src_num, dst_num = src_dst_pair

            for time_point in sorted(time_points):
                # Check if we should stop
                if stop_flag.value or time.time() >= end_time:
                    break

                # Get fresh references to hosts
                try:
                    src_host = self.hosts.get(f"h{src_num}")
                    dst_host = self.hosts.get(f"h{dst_num}")

                    if (
                        not src_host
                        or not dst_host
                        or not src_host.intf()
                        or not dst_host.intf()
                    ):
                        continue

                    scaled_time_point = time_point / time_scale
                    time_to_wait = scaled_time_point - (time.time() - start_time)

                    if time_to_wait > 0:
                        time.sleep(time_to_wait)

                    current_hour = (time_point / 3600) % self.period_hours
                    if not reset_flag.value:
                        self.start_iperf_flow(src_host, dst_host, current_hour)
                except Exception as e:
                    info(f"*** Error in flow {src_num}->{dst_num}: {str(e)}\n")
                    continue
        except Exception as e:
            info(f"*** Error in Poisson process: {str(e)}\n")

    def start_iperf_flow(self, src_host, dst_host, current_hour):
        """Modified iperf flow with variable duration and bandwidth based on domain"""
        try:
            if not src_host.intf() or not dst_host.intf():
                return

            port = random.randint(5000, 6000)

            # Determine flow duration and bandwidth based on source domain
            src_num = int(src_host.name[1:])  # Extract number from host name

            if src_num <= 7:  # Domain 1
                duration = self.flow_duration * 2  # Medium duration
                bw = random.uniform(15, 25)  # Medium-high bandwidth
            elif src_num <= 13:  # Domain 2
                duration = self.flow_duration  # Normal duration
                bw = random.uniform(8, 12)  # Low bandwidth
            elif src_num <= 19:  # Domain 3
                duration = int(self.flow_duration * 0.5)  # Short duration
                bw = random.uniform(3, 7)  # Very low bandwidth
            else:  # Domain 4
                duration = self.flow_duration * 4  # Very long duration
                bw = random.uniform(40, 60)  # Very high bandwidth

            bw_str = f"{int(bw)}M"

            # Start iperf server with extra time for cleanup
            dst_host.cmd(f"iperf -s -u -p {port} -t {duration+5} > /dev/null 2>&1 &")

            # Start iperf client with variable duration
            src_host.cmd(
                f"iperf -c {dst_host.IP()} -u -p {port} -t {duration} -b {bw_str} > /dev/null 2>&1 &"
            )

            info(
                f"  Flow at {current_hour:.2f}h: {src_host.name} -> {dst_host.name} ({bw_str}, {duration}s)\n"
            )
        except Exception as e:
            info(f"*** Error starting iperf flow: {str(e)}\n")


if __name__ == "__main__":

    # command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--global_controller_ip", type=str, default="192.168.2.33")
    parser.add_argument("--period_hours", type=float, default=24.0)
    parser.add_argument("--total_hours", type=float, default=24.0)
    parser.add_argument("--flow_duration", type=int, default=10)
    parser.add_argument("--time_scale", type=float, default=60.0)
    args = parser.parse_args()

    topology = JanosUSTopology(args)
