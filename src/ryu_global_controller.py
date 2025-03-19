from ryu.base import app_manager
from ryu.controller import ofp_event
from ryu.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from ryu.lib import hub
from ryu.ofproto import ofproto_v1_3
from ryu import cfg
from flask import Flask, request, jsonify
import urllib.request
import urllib.error
import json
import subprocess
import numpy as np
import time


class GlobalController(app_manager.RyuApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super(GlobalController, self).__init__(*args, **kwargs)

        # load the command line arguments.
        self.CONF = cfg.CONF

        # Register some of the configuration options.
        self.CONF.register_opts(
            [
                cfg.IntOpt(
                    "total_switches",
                    default=-1,
                    help=("The total number of switches in the network"),
                ),
                cfg.IntOpt(
                    "num_controllers",
                    default=-1,
                    help=(
                        "The number of controllers that this global controller manages"
                    ),
                ),
                cfg.ListOpt(
                    "capacities",
                    default=None,
                    help=(
                        "A list of capacities for each controller (ordered by IP then port)"
                    ),
                ),
            ]
        )

        # TODO add a dynamic way to get this:
        self.switches_by_controller = [
            [0],  # first controller's switches (owns switch 1)
            [1],  # second controller's switches (owns switch 2)
        ]

        self.m = self.CONF.num_controllers
        self.n = self.CONF.total_switches

        self.capacities = np.array(self.CONF.capacities, dtype=int)

        # # load the capacities for each controller from config file
        # for i in range(1,self.CONF.num_controllers+1):
        #     self.CONF.register_opts([
        #         cfg.IntOpt(f"c{i}_capacity", default=-1, help=(f"The capacity of controller {i}"))
        #     ])

        # List of domain controllers to poll
        self.domain_controllers = [
            # {'ip': '127.0.0.1', 'port': 7000}  # Example controller 1
            # {'ip': '127.0.0.1', 'port': 6655}   # Example controller 2
        ]

        self.state_matrix = np.zeros(
            (self.m, self.n)
        )  # m x n matrix, where m is the number of controllers, and n is the number of switches

        # set up flask server that will handle the registration of the domain controllers
        self.app = Flask(__name__)

        @self.app.route("/register", methods=["POST"])
        def register_domain_controller():
            print("received register request")
            # save the information

            try:
                # parse the incoming json data
                data = request.get_json()

                # check if the data is valid
                if not data:
                    return (
                        jsonify({"error": "No data provided"}),
                        400,
                    )  # Bad Request if no data

                # You can add your processing logic here
                # For example, checking required fields:
                if "controller_ip" not in data or "flask_port" not in data:
                    return (
                        jsonify({"error": "missing required fields"}),
                        400,
                    )  # Bad Request if missing required fields

                self.domain_controllers.append(
                    {"ip": data["controller_ip"], "port": data["flask_port"]}
                )

                # sorting the domain controllers, so that the order is constant, (important for building state vector)
                # Sort the domain controllers by IP and then by port
                self.domain_controllers.sort(key=lambda x: (x["ip"], x["port"]))
                print("updated domain controllers: ", self.domain_controllers)

                # Return a success response with a 201 status code (Created) if successful
                return "", 200
            except Exception as e:
                # handle unexpected errors:
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @self.app.route("/migrate", methods=["POST"])
        def migrate_switch():
            """
            This function allows the global controller to migrate a switch.
            It is intended to be called by the pytorch code via http

            it should pass two arguments in the request:
            controller_id: the index of the controller migrated to (sorted by IP, then PORT)
            switch_id: the switch that is being migrated to that controller.
            """
            try:
                # parse the incoming json data
                data = request.get_json()

                # check if the data is valid
                if not data:
                    return (
                        jsonify({"error": "No data provided"}),
                        400,
                    )  # Bad Request if no data

                # You can add your processing logic here
                # For example, checking required fields:
                if "target_controller" not in data or "switch" not in data:
                    return (
                        jsonify({"error": "missing required fields"}),
                        400,
                    )  # Bad Request if missing required field

                # execute the migration command

                # get the current controller ip
                ip = self.domain_controllers[data["target_controller"]]["ip"]
                port = self.domain_controllers[data["target_controller"]]["port"]
                subprocess.Popen(
                    f'sudo ovs-vsctl set-controller s{data["switch"]} tcp:{ip}:{port}',
                    shell=True,
                )

                # Return a success response with a 201 status code (Created) if successful
                return "", 200
            except Exception as e:
                # handle unexpected errors:
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @self.app.route("/state", methods=["GET"])
        def get_state():
            """
            This function reports the state.
            It is intended to be called by the pytorch code via http
            """
            try:
                # update the state matrix first
                self.poll_domain_controllers()
                data = jsonify({"state": self.state_matrix.tolist()})

                # Return a success response with a 201 status code (Created) if successful
                return data, 200
            except Exception as e:
                # handle unexpected errors:
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @self.app.route("/capacities", methods=["GET"])
        def get_capacities():
            """
            This function reports the capacities of the controllers.
            It is intended to be called by the pytorch code via http
            """
            print("reporting capacities to deep learning system")
            try:
                # TODO update the state matrix first, instead of polling regularly.
                data = jsonify({"data": self.capacities.tolist()})

                # Return a success response with a 201 status code (Created) if successful
                return data, 200
            except Exception as e:
                # handle unexpected errors:
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        @self.app.route("/switches_by_controller", methods=["GET"])
        def get_switch_by_controller():
            """
            This function reports which controller owns each switch.
            It is intended to be called by the pytorch code via http
            """
            print("reporting switch configuration to deep learning system")
            try:
                # TODO update the state matrix first, instead of polling regularly.
                data = jsonify({"data": self.switches_by_controller})

                # Return a success response with a 201 status code (Created) if successful
                return data, 200
            except Exception as e:
                # handle unexpected errors:
                return (
                    jsonify({"error": f"Internal server error: {str(e)}"}),
                    500,
                )  # Internal Server Error on exceptions

        # # testing the switch migration code
        # hub.sleep(1)
        # subprocess.Popen(f'sudo ovs-vsctl set-controller s1 tcp:127.0.0.1:6654', shell=True)
        # subprocess.Popen(f'sudo ovs-vsctl set-controller s2 tcp:127.0.0.1:6654', shell=True)

        self.flask_port = 8000
        self.flask_thread = hub.spawn(self.run_flask)

        # now that we have registered the controllers, we can begin polling for network traffic information.

        # # Polling interval in seconds
        # self.poll_interval = 1
        # # Periodic polling using greenlet
        # self.poll_thread = hub.spawn(self.poll_domain_controllers)

    def run_flask(self):
        self.app.run(host="0.0.0.0", port=self.flask_port)

    def poll_domain_controllers(self):
        """
        Polls the domain controllers for their state, then sets the hub to sleep for the poll interval.
        """
        # Reset state matrix
        self.state_matrix = np.zeros((self.m, self.n))

        # Poll each controller for its state
        for i, controller in enumerate(self.domain_controllers):
            controller_state = self._get_controller_state(
                controller["ip"], controller["port"]
            )

            if controller_state:
                # Extract packet-in rates from the controller state
                packet_in_rates = controller_state.get("packet_in_rates", {})

                # Convert to state vector
                state_vector = [0] * self.n
                for switch_id_str, rate in packet_in_rates.items():
                    try:
                        # Convert string switch ID to integer and adjust for 0-indexing
                        switch_idx = int(switch_id_str) - 1
                        if 0 <= switch_idx < self.n:
                            state_vector[switch_idx] = rate
                    except (ValueError, IndexError):
                        self.logger.error(f"Invalid switch ID: {switch_id_str}")

                # Update state matrix
                self.state_matrix[i, :] = state_vector

        # Combine the results from the domain controllers into a single state vector
        # size m x n where m is the number of domain controllers, n is the number of switches.
        print("state_matrix: ", self.state_matrix)


    def _get_controller_state(self, ip, port):
        """
        Makes a REST API call to the /controller_state endpoint of the domain controllers
        """
        url = f"http://{ip}:{port}/controller_state"
        try:
            # Create request object
            req = urllib.request.Request(url)

            # Send the request with a timeout
            with urllib.request.urlopen(req, timeout=2) as response:
                response_code = response.getcode()

                if response_code == 200:
                    # Read and parse the response
                    response_data = response.read().decode("utf-8")
                    controller_state = json.loads(response_data)
                    return controller_state
                else:
                    self.logger.error(
                        f"Failed to get state from {ip}:{port}. Status code: {response_code}"
                    )
                    return None
        except urllib.error.URLError as e:
            self.logger.error(f"Error while requesting state from {ip}:{port}: {e}")
            return None
        except urllib.error.HTTPError as e:
            self.logger.error(
                f"HTTP Error while requesting state from {ip}:{port}: {e.code} {e.reason}"
            )
            return None
        except Exception as e:
            self.logger.error(
                f"Unexpected error while requesting state from {ip}:{port}: {e}"
            )
            return None

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, MAIN_DISPATCHER)
    def _switch_features_handler(self, ev):
        self.logger.info(f"Switch {ev.switch.dp.id} is connected")
