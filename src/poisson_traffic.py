import numpy as np
import time
from scapy.all import sendp, Ether, IP, UDP
import argparse

# packages that you need here must be installed globally on your system
# or you won't have access to them inside the hosts. 

print("poisson_traffic called")

# set up the argument parser
parser = argparse.ArgumentParser(description="Generate Poisson Traffic")

# define the arguments
parser.add_argument('--dst_ip', type=str, required=True, help="Destination IP address")
parser.add_argument('--dst_port', type=int, required=True, help="Destination port")
parser.add_argument('--lambda_rate', type=float, required=True, help="Poisson rate (packets/sec)")
parser.add_argument('--iface', type=str, required=True, help="interface")


# Parse the arguments
args = parser.parse_args()

# Assign arguments to variables
dst_ip = args.dst_ip
dst_port = args.dst_port
lambda_rate = args.lambda_rate

iface = args.iface

# Traffic generation loop
while True:
    inter_arrival_time = np.random.exponential(1 / lambda_rate)
    time.sleep(inter_arrival_time)
    pkt = Ether() / IP(dst=dst_ip) / UDP(dport=dst_port)
    sendp(pkt, iface=iface, verbose=False)