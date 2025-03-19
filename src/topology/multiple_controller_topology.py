#!/usr/bin/env python3

from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
import time, sys

from mininet.util import pmonitor

HOST_IP = "127.0.0.1" # this will depend on the setup # TODO make this an argument/parameter
 
def create_topology():
    """
    in this topology, we have 3 controllers, 2 switches, and 2 hosts
    there is a global controller that communicates with only the controllers to get traffic information
    each of the other two controllers only communicates with one switch
    each switch manages one host. 
    """
    net = Mininet(topo=None, build=False)
    
    # Add controllers (1 global + 4 domain controllers)
    info("*** Adding controllers\n")
    # each controller uses the local HOST_IP, but uses a different port. 

    # Global Controller

    #TODO add Global Controller and figure how how the connection works. 
    c0 = net.addController("g0", controller=RemoteController, ip=HOST_IP, port=6653)

    # Domain Controllers
    c1 = net.addController('c1', controller=RemoteController, ip=HOST_IP, port=6654)
    c2 = net.addController('c2', controller=RemoteController, ip=HOST_IP, port=6655)

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

    # Start network
    info("*** Starting network\n")
    net.build()
    c1.start()
    c2.start()
    c0.start()
    s1.start([c1])
    s2.start([c2])

    # Wait for controller to initialize
    info("*** Waiting for controller initialization (3 seconds)...\n")
    time.sleep(3)

    script_path = "./poisson_traffic.py"
        
    popens = {}
    # for host in net.hosts[-1]

    # for host in net.hosts:
    #     popens[host] = host.popen("ping -c5 %s" % last.IP() )
    #     last = host
    
    # popens[h1] = h1.popen(f"python3 {script_path} &", shell=True)
    popens[h2] = h2.popen(f"python3 {script_path} --dst_ip 127.0.0.1 --dst_port 6655 --iface h2-eth0 --lambda_rate 50 &", shell=True)
    popens[h1] = h1.popen(f"python3 {script_path} --dst_ip 127.0.0.1 --dst_port 6654 --iface h1-eth0 --lambda_rate 50 &", shell=True)

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
