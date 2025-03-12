#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time, sys

from mininet.util import pmonitor

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
    info("*** Waiting for controller initialization (3 seconds)...\n")
    time.sleep(3)

    # # Test connectivity
    # info("*** Testing connectivity\n")
    # info("*** Ping: h1 -> h2\n")
    # result = h1.cmd("ping -c 3 %s" % h2.IP())
    # info(result + "\n")


    script_path = "./poisson_traffic.py"
    # Run the poisson traffic generating script inside h1
    # popen allows you to run a command inside the host.
    # & is used to that it runs in the background and doesn't block. 
    # shell = True allows for executing shell commands. 


    # https://github.com/mininet/mininet/blob/master/examples/popen.py 
        
    popens = {}
    # for host in net.hosts[-1]

    # for host in net.hosts:
    #     popens[host] = host.popen("ping -c5 %s" % last.IP() )
    #     last = host
    
    # popens[h1] = h1.popen(f"python3 {script_path} &", shell=True)
    popens[h1] = h1.popen(f"python3 {script_path} --dst_ip 127.0.0.1 --dst_port 6653 --iface h1-eth0 --lambda_rate 50 &", shell=True)
    popens[h2] = h2.popen(f"python3 {script_path} --dst_ip 127.0.0.1 --dst_port 6653 --iface h2-eth0 --lambda_rate 50 &", shell=True)

    # monitor them and print output
    try:
        for host, line in pmonitor(popens):
            if host:
                info( "<%s>: %s" % ( host.name, line ) )
    except KeyboardInterrupt:
        print("bye")
        net.stop()
        sys.exit()

    # Start CLI
    info("*** Running CLI\n")
    CLI(net)

    # Stop network
    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
