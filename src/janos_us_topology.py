#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink

HOST_IP = "192.168.2.33"

def create_topology():
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
        switches["s{}".format(i)] = net.addSwitch("s{}".format(i), stp=True, failMode="standalone")

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

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
