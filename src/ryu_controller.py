# Copyright (C) 2016 Nippon Telegraph and Telephone Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu import cfg
from ryu.lib import hub
from flask import Flask, jsonify
import os
import requests
import sys
import time # used for the flow rate calculation.

class ExampleSwitch13(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(ExampleSwitch13, self).__init__(*args, **kwargs)
        # initialize mac address table.
        self.mac_to_port = {}

        # Define a command-line argument
        self.CONF = cfg.CONF

        # Register configuration option for listen port
        self.CONF.register_opts([
            cfg.IntOpt("total_switches", default=-1, help=("The total number of switches in the network")),
            cfg.StrOpt("global_controller_ip", default="127.0.0.1", help=("The global controller's ip address")),
            cfg.StrOpt("global_controller_flask_port", default="-1", help=("The port of the flask endpoint running on global controller"))      
        ])

        
        # Access the listen port
        self.listen_port = self.CONF.ofp_tcp_listen_port
        self.flask_port = self.listen_port + 500 # assign a new port to use for communication with the global controller.
        
        # print("listen_port: ", self.listen_port)

        # send message to the global controller to register


        print("total_switches: ", self.CONF.total_switches)

        # # for tracking in-messages for deep learning state vector
        # self.total_switches = int(kwargs.get('total_switches', -1))
        # if self.total_switches < 0:
        #     raise Exception("argument missing: total_switches. This should be the total number of switches in your network")

        # self.in_flows = [0]*self.total_switches # the number of packet flows in from 

        # Read the argument value
        self.total_switches = self.CONF.total_switches
        # Initialize an array to track in-messages per switch
        self.in_flows = [0] * self.total_switches 

        self.register_with_global_controller()

        self.app = Flask(__name__)

        @self.app.route('/get_state', methods=["GET"])
        def get_state():
            # return the in flows (state vector) #TODO modify this to use flow rate instead
            # compute the time difference
            timediff_s = time.perf_counter() - self.start_time
            # divide each by time to get rate 
            self.in_rate = [x / timediff_s for x in self.in_flows]
            # reset the in_flows and time
            self.in_flows = [0] * self.total_switches 
            self.start_time = time.perf_counter()
            # send to the global controller
            return jsonify(state_vector=self.in_rate)

        self.flask_thread = hub.spawn(self.run_flask)

        self.start_time = time.perf_counter()  # High-precision timer
        

    def register_with_global_controller(self):
        global_controller_ip = self.CONF.global_controller_ip
        global_controller_flask_port = self.CONF.global_controller_flask_port
        global_controller_url = f"http://{global_controller_ip}:{global_controller_flask_port}/register"

        # define the payload
        data = {
            "controller_ip": "127.0.0.1",  # The domain controller's IP
            # "controller_port": self.listen_port,
            "flask_port": self.flask_port
        }

        while True:
            try:
                # Send a POST request to the global controller to register
                response = requests.post(global_controller_url, json=data)
                if response.status_code == 200:
                    self.logger.info(f"Successfully registered with the global controller: {response.text}")
                    return
                else:
                    self.logger.error(f"Failed to register with global controller, status code: {response.status_code}")
                    raise Exception("Failed to register with the global controller")
            except requests.exceptions.RequestException as e:
                self.logger.error(f"Error while trying to connect to the global controller: {e}")
            
            # Wait for a while before sending the next "hello" message (e.g., every 30 seconds)
            hub.sleep(30)

    def run_flask(self):
        self.app.run(host="127.0.0.1", port=self.flask_port)
    
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install the table-miss flow entry.
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER,
                                          ofproto.OFPCML_NO_BUFFER)]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # construct flow_mod message and send it.
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS,
                                             actions)]
        mod = parser.OFPFlowMod(datapath=datapath, priority=priority,
                                match=match, instructions=inst)
        datapath.send_msg(mod)

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def _packet_in_handler(self, ev):
        
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # get Datapath ID to identify OpenFlow switches.
        dpid = datapath.id
        self.mac_to_port.setdefault(dpid, {})

        # analyse the received packets using the packet library.
        pkt = packet.Packet(msg.data)
        eth_pkt = pkt.get_protocol(ethernet.ethernet)
        dst = eth_pkt.dst
        src = eth_pkt.src

        # get the received port number from packet_in message.
        in_port = msg.match['in_port']

        # self.logger.info("packet in %s %s %s %s", dpid, src, dst, in_port)
        
        # update in_flows
        print(f"packet in from {dpid}")
        self.in_flows[dpid-1] += 1
        
        # learn a mac address to avoid FLOOD next time.
        self.mac_to_port[dpid][src] = in_port

        # if the destination mac address is already learned,
        # decide which port to output the packet, otherwise FLOOD.
        if dst in self.mac_to_port[dpid]:
            out_port = self.mac_to_port[dpid][dst]
        else:
            out_port = ofproto.OFPP_FLOOD

        # construct action list.
        actions = [parser.OFPActionOutput(out_port)]

        # install a flow to avoid packet_in next time.
        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst)
            self.add_flow(datapath, 1, match, actions)

        # construct packet_out message and send it.
        out = parser.OFPPacketOut(datapath=datapath,
                                  buffer_id=ofproto.OFP_NO_BUFFER,
                                  in_port=in_port, actions=actions,
                                  data=msg.data)
        datapath.send_msg(out)