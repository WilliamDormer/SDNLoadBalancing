#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time


def create_topology():
    """Create a simple topology with just two switches connected directly"""
    net = Mininet(topo=None, build=False)

    # Add controller
    info("*** Adding controller\n")
    c0 = net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)

    # Add switches
    info("*** Adding switches\n")
    s1 = net.addSwitch("s1")
    s2 = net.addSwitch("s2")

    # Add hosts
    info("*** Adding hosts\n")
    h1 = net.addHost("h1", ip="10.0.0.1/8", mac="00:00:00:00:00:01")
    h2 = net.addHost("h2", ip="10.0.0.2/8", mac="00:00:00:00:00:02")

    # Add links
    info("*** Adding links\n")
    net.addLink(h1, s1)
    net.addLink(h2, s2)
    net.addLink(s1, s2)  # Direct link between switches

    # Start network
    info("*** Starting network\n")
    net.build()
    c0.start()
    s1.start([c0])
    s2.start([c0])

    # Wait for controller to initialize
    info("*** Waiting for controller initialization (5 seconds)...\n")
    time.sleep(5)

    # Test connectivity
    info("*** Testing connectivity\n")
    info("*** Ping: h1 -> h2\n")
    result = h1.cmd("ping -c 3 %s" % h2.IP())
    info(result + "\n")

    # Start CLI
    info("*** Running CLI\n")
    CLI(net)

    # Stop network
    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
