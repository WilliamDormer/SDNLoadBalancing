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
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, DEAD_DISPATCHER
from ryu.controller.handler import set_ev_cls
from ryu.ofproto import ofproto_v1_3
from ryu.lib.packet import packet
from ryu.lib.packet import ethernet
from ryu import cfg
from ryu.lib import hub
from flask import Flask, jsonify, request
import urllib.request
import urllib.error
import json
import time  # used for the flow rate calculation.
import uuid


class DomainController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(DomainController, self).__init__(*args, **kwargs)
        # initialize mac address table.
        self.mac_to_port = {}

        # Define a command-line argument
        self.CONF = cfg.CONF

        # Register configuration option for listen port
        self.CONF.register_opts(
            [
                cfg.IntOpt(
                    "total_switches",
                    default=-1,
                    help=("The total number of switches in the network"),
                ),
                cfg.StrOpt(
                    "global_controller_ip",
                    default="127.0.0.1",
                    help=("The global controller's ip address"),
                ),
                cfg.StrOpt(
                    "global_controller_flask_port",
                    default="8000",
                    help=(
                        "The port of the flask endpoint running on global controller"
                    ),
                ),
                cfg.StrOpt(
                    "controller_id",
                    default="",
                    help=("Unique identifier for this controller"),
                ),
            ]
        )

        # Access the listen port
        self.listen_port = self.CONF.ofp_tcp_listen_port

        # Generate or use controller ID
        if self.CONF.controller_id:
            self.controller_id = self.CONF.controller_id
        else:
            self.controller_id = f"controller-{self.listen_port}"

        # Read the argument value
        self.total_switches = self.CONF.total_switches

        # State tracking components
        self.datapaths = {}  # Store active datapaths
        self.controller_switch_mapping = (
            {}
        )  # g_hi(t) - maps switch DPIDs to controller ID
        self.in_flows = [0] * self.total_switches  # Packet-in counter for each switch
        self.in_rate = [0] * self.total_switches  # Packet-in rate for each switch
        self.controller_load = 0  # Overall controller load
        self.start_time = time.perf_counter()  # High-precision timer

        # Set up Flask server for communication with global controller
        self.flask_port = self.listen_port + 500  # Use a different port for Flask
        self.app = Flask(__name__)

        # Define Flask routes

        @self.app.route("/controller_state", methods=["GET"])
        def get_controller_state():
            """
            Consolidated endpoint that returns all controller state information
            """
            # Calculate packet-in rates
            self.calculate_packet_in_rates()

            # Calculate controller load
            self.calculate_controller_load()

            state = {
                "controller_id": self.controller_id,
                "controller_load": self.controller_load,
                "controller_switch_mapping": {
                    str(k): v for k, v in self.controller_switch_mapping.items()
                },
                "packet_in_rates": {
                    str(dpid): self.in_rate[dpid - 1]
                    for dpid in self.controller_switch_mapping
                },
            }
            return jsonify(state)

        # Start Flask server in a separate thread
        self.flask_thread = hub.spawn(self.run_flask)

        # Register with global controller
        self.register_with_global_controller()


    def run_flask(self):
        """
        Run the Flask server
        """
        self.app.run(host="127.0.0.1", port=self.flask_port)

    def register_with_global_controller(self):
        global_controller_ip = self.CONF.global_controller_ip
        global_controller_flask_port = self.CONF.global_controller_flask_port
        global_controller_url = (
            f"http://{global_controller_ip}:{global_controller_flask_port}/register"
        )

        # define the payload
        data = {
            "controller_id": self.controller_id,
            "controller_ip": "127.0.0.1",  # The domain controller's IP
            "controller_port": self.listen_port,
            "flask_port": self.flask_port,  # Add the Flask port for communication
        }

        while True:
            try:
                # Convert data to JSON and encode as bytes
                json_data = json.dumps(data).encode("utf-8")

                # Create request object
                req = urllib.request.Request(
                    global_controller_url,
                    data=json_data,
                    headers={"Content-Type": "application/json"},
                )

                # Send the request
                with urllib.request.urlopen(req) as response:
                    response_text = response.read().decode("utf-8")
                    response_code = response.getcode()

                    if response_code == 200:
                        self.logger.info(
                            f"Successfully registered with the global controller: {response_text}"
                        )
                        return
                    else:
                        self.logger.error(
                            f"Failed to register with global controller, status code: {response_code}"
                        )
                        raise Exception("Failed to register with the global controller")
            except urllib.error.HTTPError as e:
                self.logger.error(
                    f"HTTP Error while trying to connect to the global controller: {e.code} {e.reason}"
                )
            except Exception as e:
                self.logger.error(
                    f"Unexpected error while trying to connect to the global controller: {e}"
                )

            # Wait for a while before sending the next "hello" message (e.g., every 30 seconds)
            hub.sleep(30)

    def calculate_packet_in_rates(self):
        """
        Calculate packet-in rates for all switches
        """
        timediff_s = time.perf_counter() - self.start_time
        if timediff_s > 0:
            # Calculate rates
            self.in_rate = [x / timediff_s for x in self.in_flows]
            # Reset counters
            self.in_flows = [0] * self.total_switches
            self.start_time = time.perf_counter()

    def calculate_controller_load(self):
        """
        Calculate overall controller load
        """
        # For simplicity, use sum of packet-in rates as controller load
        self.controller_load = sum(self.in_rate)

    @set_ev_cls(ofp_event.EventOFPStateChange, [MAIN_DISPATCHER, DEAD_DISPATCHER])
    def state_change_handler(self, ev):
        """
        Handle switch connect/disconnect events
        """
        datapath = ev.datapath
        if ev.state == MAIN_DISPATCHER:
            if datapath.id not in self.datapaths:
                self.logger.info(f"Switch {datapath.id} connected")
                self.datapaths[datapath.id] = datapath
                # Update controller-switch mapping
                self.controller_switch_mapping[datapath.id] = self.controller_id
        elif ev.state == DEAD_DISPATCHER:
            if datapath.id in self.datapaths:
                self.logger.info(f"Switch {datapath.id} disconnected")
                del self.datapaths[datapath.id]
                # Update controller-switch mapping
                if datapath.id in self.controller_switch_mapping:
                    del self.controller_switch_mapping[datapath.id]

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # install the table-miss flow entry.S
        match = parser.OFPMatch()
        actions = [
            parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)
        ]
        self.add_flow(datapath, 0, match, actions)

    def add_flow(self, datapath, priority, match, actions):
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # construct flow_mod message and send it.
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath, priority=priority, match=match, instructions=inst
        )
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
        in_port = msg.match["in_port"]

        # self.logger.debug("packet in %s %s %s %s", dpid, src, dst, in_port)

        # update in_flows
        if dpid <= self.total_switches:
            self.in_flows[dpid - 1] += 1

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
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=ofproto.OFP_NO_BUFFER,
            in_port=in_port,
            actions=actions,
            data=msg.data,
        )
        datapath.send_msg(out)
