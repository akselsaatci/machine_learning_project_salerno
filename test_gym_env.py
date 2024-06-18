import gym
from test_pybullet_env import Playfield, JOINTS, STATE_DIMENSION, TABLE_LENGTH, TABLE_WIDTH, TABLE_HEIGHT, BALL_SERVICE_HEIGHT
from gym import spaces
import numpy as np


class PlayfieldEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(PlayfieldEnv, self).__init__()
        self.next_state = None

        self.pf = Playfield(gui=True)

        # Define action and observation space
        # Assuming action space is continuous with bounds [-1, 1] for each joint
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(JOINTS,), dtype=np.float32)

        # Assuming observation space is continuous and consists of the state dimension
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSION,), dtype=np.float32)
        # Initialize the target coordinate
        self.target_coordinate = self._generate_random_target()

    def _generate_random_target(self):
        # Generate a random coordinate within a specific range between the table and robot0
        robot0_pos = [0, -0.85 - 0.5 * TABLE_LENGTH, 0.8]
        table_pos = [0, 0, TABLE_HEIGHT]

        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1.5, -1.5)
        z = np.random.uniform(2, 2.5)

        return np.array([x, y, z])

    def reset(self):
        # Reset the simulation to initial state

        self.pf.reset()

        self.pf.schedule_start_positions()

        self.target_coordinate = self._generate_random_target()

        print(f"Target coordinate: {self.target_coordinate}")
        # Add a box to the environment
        self.pf.add_box(position=self.target_coordinate)

        # Return the initial observation
        return self._get_observation()

    def step(self, action):
        # Set the robot joints according to the action
        self.pf.set_robot_joints(0, action)

        # Step the simulation
        self.pf.step_simulation()
        self.pf.update()

        # Get the new observation
        observation = self._get_observation()

        # Compute reward
        reward = self._compute_reward()

        # Check if episode is done
        done = self._is_done()

        # Additional info
        info = {}

        return observation, reward, done, info

        # Define a callback function to update the state of the environment

    def render(self, mode='human'):
        # Render the simulation if needed (handled by Playfield)
        pass

    def _get_observation(self):
        # Construct observation from the Playfield state
        # For simplicity, we'll just use the robot's joint positions for now
        obs = np.array(self.pf.get_robot_joints(0))
        return obs

    def _compute_reward(self):
        ball_position = np.array(self.pf.ball_position)
        distance_to_target = np.linalg.norm(
            ball_position - self.target_coordinate)
        reward = -distance_to_target
        return reward

    def _is_done(self):
        # Check if the robot's paddle has reached the target coordinate
        paddle_position, _ = self.pf.get_paddle_position_and_normal(0)
        distance_to_target = np.linalg.norm(
            np.array(paddle_position) - self.target_coordinate)

        # Define a threshold distance to consider the task done
        threshold_distance = 0.1

        if distance_to_target < threshold_distance:
            return True

        return False


# To use this environment
if __name__ == '__main__':
    env = PlayfieldEnv()
    obs = env.reset()
    for _ in range(100000):
        action = env.action_space.sample()  # Take a random action
        obs, reward, done, info = env.step(action)
        if done:
            break
    env.close()
