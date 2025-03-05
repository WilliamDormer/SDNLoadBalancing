from mininet.net import Mininet
from mininet.node import Controller, RemoteController
from mininet.cli import CLI
from mininet.log import setLogLevel, info
from mininet.link import TCLink
from mininet.node import OVSKernelSwitch

HOST_IP = "192.168.2.33"


def create_topology():
    net = Mininet(topo=None, build=False, ipBase="10.0.0.0/8")

    info("*** Adding controller\n")
    c0 = net.addController(
        name="c0", controller=RemoteController, ip=HOST_IP, port=6653
    )

    # Add switches
    info("*** Adding switches\n")
    switches = {}
    for i in range(1, 4):
        switches["v{}".format(i)] = net.addSwitch("v{}".format(i), stp=True, failMode="standalone")

    # Add hosts
    hosts = {}
    for i in range(1, 4):
        hosts["h{}".format(i)] = net.addHost(
            "h{}".format(i), mac="00:00:00:00:00:{}".format(i), ip="10.0.0.{}".format(i)
        )

    # Add links
    for i in range(1, 4):
        net.addLink(hosts["h{}".format(i)], switches["v{}".format(i)])

    # Add links
    # links = [
    # # Domain 1 internal links
    #     ("v1", "v2"),
    #     ("v1", "v5"),
    #     ("v2", "v3"),
    #     ("v2", "v5"),
    #     ("v3", "v6"),
    #     ("v3", "v4"),
    #     ("v4", "v5"),
    #     ("v4", "v6"),
    #     ("v5", "v7"),
    # ]

    net.addLink(switches["v1"], switches["v2"])
    net.addLink(switches["v2"], switches["v3"])
    net.addLink(switches["v3"], switches["v1"])
    # Start network
    info("*** Starting network\n")
    net.build()
    c0.start()
    for i in range(1, 4):
        switches["v{}".format(i)].start([c0])

    info("*** Running CLI\n")
    CLI(net)

    info("*** Stopping network\n")
    net.stop()


if __name__ == "__main__":
    setLogLevel("info")
    create_topology()
