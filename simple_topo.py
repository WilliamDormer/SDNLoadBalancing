#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink


def create_topology():
    # Create a network with a remote controller
    net = Mininet(controller=RemoteController, link=TCLink)

    # Add controller
    info("*** Adding controller\n")
    c0 = net.addController("c0", controller=RemoteController, ip="127.0.0.1", port=6653)

    # Add switches
    info("*** Adding switches\n")
    s1 = net.addSwitch("s1")
    s2 = net.addSwitch("s2")

    # Add hosts
    info("*** Adding hosts\n")
    h1 = net.addHost("h1", mac="00:00:00:00:00:01", ip="10.0.0.1/24")
    h2 = net.addHost("h2", mac="00:00:00:00:00:02", ip="10.0.0.2/24")
    h3 = net.addHost("h3", mac="00:00:00:00:00:03", ip="10.0.0.3/24")
    h4 = net.addHost("h4", mac="00:00:00:00:00:04", ip="10.0.0.4/24")

    # Add links
    info("*** Creating links\n")
    net.addLink(h1, s1)
    net.addLink(h2, s1)
    net.addLink(s1, s2)
    net.addLink(h3, s2)
    net.addLink(h4, s2)

    # Start network
    info("*** Starting network\n")
    net.build()
    c0.start()
    s1.start([c0])
    s2.start([c0])

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
