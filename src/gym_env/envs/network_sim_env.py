import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests
import time
import pygame


class NetworkSimEnv(gym.Env):
    # Human render mode will show what controllers the switches belong to in real time.
    # None will simply run with no GUI.
    metadata = {"render_modes": ["human", "no"], "render_fps": 4}

    def __init__(
        self,
        num_controllers,
        num_switches,
        max_rate,
        gc_ip,
        gc_port,
        step_time,
        reset_timeout=30,
        window_size=512,
        render_mode=None,
    ):
        """
        num_controllers: the number of controllers
        num_switches: the number of switches
        max_rate: used as the max value of the observation space box.
        gc_ip: a string holding the IP address of the global controller
        gc_port: a string holding the flask application port on the global controller
        step_time: a float indicating the number of second to wait after executing a migration action to wait before reporting the reward and observations etc.
        window_size: the size of the pygame window for human rendering.
        """
        self.m = num_controllers
        self.n = num_switches
        self.max_rate = max_rate
        self.gc_ip = gc_ip
        self.gc_port = gc_port
        self.step_time = step_time
        self.window_size = window_size
        self.reset_timeout = reset_timeout

        self.gc_base_url = f"http://{gc_ip}:{self.gc_port}/"

        self.D_t_prev = None  # this stores the previous iteration's value for D_t, which is used in the reward calculation.

        # define the observation space

        # the system state is defined by the matrix St, which is an m x n matrix, storing the message request rates.
        # TODO why do they use the message rate instead of the load ratio for the observation?
        self.observation_space = spaces.Box(
            low=0, high=max_rate, shape=(self.m, self.n), dtype=np.float32
        )

        # define the action space, 1, 2, ... mxn
        # move switch to controller, of which n of them are non-migration actions.
        self.action_space = spaces.Discrete(n=self.m * self.n, start=1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # get the capacities of each controller from the network.
        self.capacities = self.get_capacities()

        self.switches_by_controller = self._get_switches_by_controller()

        """
        If human-rendering is used 'self.window' will be a reference to the window that 
        we draw to. 'self.clock' will be a clock that is used to ensure that the environmet
        is rendered at the correct framerate in human mode. They will remain "none" until human-mode
        is used for the first time
        """
        self.window = None
        self.clock = None

        # Add tracking for historical loads
        self.historical_loads = []

        # Initialize accumulated loads for each controller
        self.controller_accumulated_loads = np.zeros(self.m)

    def _get_switches_by_controller(self):
        """
        function that polls the global controller for the switch configuration. Should be called at initialization, then on a migrate action.
        """
        try:
            data = {}
            url = self.gc_base_url + "switches_by_controller"
            response = requests.get(url, json=data)
            if response.status_code == 200:
                # get the state and return it
                json_data = response.json()  # Convert response to a dictionary

                # update self.switches_by_controller
                # print("switches_by_controller: ", json_data)
                return json_data[
                    "data"
                ]  # this will be a list of lists, where the rows are the controllers (starting 0,1,2,3), and each holds the switches it controls (indexed starting at 1)
            else:
                raise Exception("Failed to retreive capacities from network.")
        except Exception as e:
            print("there was an error in _get_switches_by_controller: ", e)

    def get_capacities(self):
        """
        function that polls the global controller for the capacities of each controller.
        should be computed using CBench before this point.
        """
        # make a request to the network simulator to retrieve the state
        data = {}
        url = self.gc_base_url + "capacities"
        response = requests.get(url)
        if response.status_code == 200:
            # get the state and return it
            json_data = response.json()  # Convert response to a dictionary
            array = np.array(json_data["data"])  # Convert list back to NumPy array
            return array
        else:
            raise Exception("Failed to retreive capacities from network.")

    def _get_obs(self):
        """
        helper function to get an observation from the environment.
        """

        # make a request to the network simulator to retrieve the state
        data = {}
        url = self.gc_base_url + "state"
        response = requests.get(url, json=data)
        if response.status_code == 200:
            # get the state and return it
            json_data = response.json()  # Convert response to a dictionary
            array = np.array(json_data["state"])  # Convert list back to NumPy array
            # convert it to the observation space
            array = np.array(array, dtype=np.float32)
            return array
        else:
            raise Exception("Failed to retreive state from network.")

    def _get_info(self):
        """
        optional method for providing data that is returend by step and reset.
        """
        info = {}
        return info

    def reset(self, seed=None, options=None):
        """
        Method called to initiate a new episode.
        Performs a synchronous reset of the environment.
        """
        super().reset(seed=seed)

        try:
            # Make synchronous reset request
            response = requests.post(
                self.gc_base_url + "reset", timeout=self.reset_timeout
            )

            if response.status_code != 200:
                raise Exception(f"Failed to reset network: {response.text}")

            # Add warm-up period to wait for initial flow statistics
            warmup_steps = 18  # Number of steps to wait for flow statistics
            for i in range(warmup_steps):
                time.sleep(self.step_time)
                # Poll for state to check if flows have started
                observation = self._get_obs()
                if (
                    np.any(observation != 0) and i > 1
                ):  # If any non-zero values are present
                    break

            if self.render_mode == "human":
                self._render_frame()

            return observation, self._get_info()
        except requests.exceptions.Timeout:
            raise Exception(
                f"Reset request timed out after {self.reset_timeout} seconds"
            )
        except Exception as e:
            raise Exception(f"Failed to reset network, error: ({e})")

    def step(self, action):
        """
        This takes the chosen action in the network (aka the switch migration decision)
        it returns:
        observation: what the new state is
        reward: what the reward value from taking that action was
        terminated: boolean indicating whether it finished successfully
        truncated: boolean indiciating if it was cut short.
        info: information.
        """

        # compute the migration action that was selected.
        # Since action space starts at 1, subtract 1 first
        action = action - 1
        # Now calculate switch (e) and controller (w)
        e = int((action % self.n) + 1)  # switch number (1-based)
        w = int(action // self.n + 1)  # controller number (1-based)

        # send the migration action to the global controller.
        data = {"target_controller": w, "switch": e}

        # print(f"migrate request data: {data}")
        # print(
        #     f"Current switch configuration before migration: {self.switches_by_controller}"
        # )

        try:
            response = requests.post(self.gc_base_url + "migrate", json=data)
            # print(f"Migration response status code: {response.status_code}")
            # print(f"Migration response content: {response.text}")

            if response.status_code != 200:
                raise Exception(
                    f"Failed to execute migration action. Status: {response.status_code}, Response: {response.text}"
                )

            # Check if the switch actually moved
            old_config = self.switches_by_controller
            # wait some time to allow the network to adjust to the change
            time.sleep(self.step_time)
            # get the updated migrated switch positions:
            self.switches_by_controller = self._get_switches_by_controller()
            # print(
            #     f"Switch configuration after migration: {self.switches_by_controller}"
            # )

        except requests.exceptions.RequestException as e:
            print(f"Network error during migration: {e}")
            raise e
        except Exception as e:
            print(f"Error during migration: {e}")
            raise e

        # get the observation
        observation = self._get_obs()
        # compute the reward

        # Compute loads and ratios
        L = np.sum(observation, axis=1)  # Controller loads
        B = L / self.capacities  # Load ratios
        B_bar = np.sum(B) / self.m  # Average load ratio

        # Compute load balancing rate (D_t) properly
        if B_bar > 0:
            D_t = np.sqrt(np.sum((B - B_bar) ** 2) / self.m) / B_bar
        else:
            D_t = 1.0  # Maximum imbalance when no load

        # Calculate reward based on load imbalance
        reward = -D_t  # Negative reward for imbalance

        # Update historical loads for better tracking
        self.historical_loads.append(L.copy())
        if len(self.historical_loads) > 10:  # Keep last 10 steps
            self.historical_loads.pop(0)

        # Check if done
        done = False
        truncated = False
        info = {"load_ratios": B, "balancing_rate": D_t, "average_load": B_bar}

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, done, truncated, info

    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):
        # use PyGame to render the network information.
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # we want to display a grid/matrix of m (number of controllers) by n(num controllers)
        # and for each switch, draw it in the correct controller spot.

        # TODO expand on this

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        pix_square_size = self.window_size / max(
            self.m, self.n
        )  # Square size based on grid dimensions

        # Draw controllers (grid cells)
        for i in range(self.m):
            for j in range(self.n):
                pygame.draw.rect(
                    canvas,
                    (200, 200, 200),
                    pygame.Rect(
                        j * pix_square_size,
                        i * pix_square_size,
                        pix_square_size,
                        pix_square_size,
                    ),
                    width=3,  # Grid lines
                )

        # Draw switches #TODO fix this rendering for the 1 indexing.
        for controller, switches in enumerate(
            self.switches_by_controller
        ):  # controller is indexed starting at 1, but the switches are not, they start at 0.
            # print("drawing for controller")
            for switch in switches:
                # print("drawing for switch")
                pygame.draw.circle(
                    canvas,
                    (0, 0, 255),  # Blue color for switches
                    (
                        ((switch - 1) + 0.5) * pix_square_size,
                        ((controller) + 0.5) * pix_square_size,
                    ),
                    pix_square_size / 4,  # Size of switch
                )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        # else:  # rgb_array
        #     return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        """
        Cleanly shut down the environment by sending stop requests to both the global controller
        and topology wrapper.
        """
        try:
            # First try to stop the global controller
            gc_url = f"{self.gc_base_url}stop"
            try:
                response = requests.post(gc_url, timeout=5)
                if response.status_code != 200:
                    print(
                        f"Warning: Failed to stop global controller: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Warning: Error stopping global controller: {e}")

            # Then try to stop the topology wrapper
            wrapper_url = "http://localhost:9000/stop_topology"
            try:
                response = requests.post(wrapper_url, timeout=5)
                if response.status_code != 200:
                    print(
                        f"Warning: Failed to stop topology wrapper: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                print(f"Warning: Error stopping topology wrapper: {e}")

            # Close pygame window if it exists
            if self.window is not None:
                import pygame

                pygame.quit()
                self.window = None
                self.clock = None

        except Exception as e:
            print(f"Warning: Error during environment cleanup: {e}")
