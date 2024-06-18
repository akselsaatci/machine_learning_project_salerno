import gym
from gym import spaces
import numpy as np
from tennis_env import Playfield
import pybullet as p
from tennis_env import JOINTS, STATE_DIMENSION


class TableTennisEnv(gym.Env):
    def __init__(self, game):
        super(TableTennisEnv, self).__init__()
        self.game = game
        self.playfield = Playfield(game, gui=True)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(JOINTS,))
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(STATE_DIMENSION,))
        self.current_state = None

    def reset(self):
        self.playfield.init_pybullet()
        self.playfield.load_objects()
        self.playfield.schedule_start_positions()
        self.current_state = self.game.compute_state(0)
        return np.array(self.current_state)

    def step(self, action):
        self.playfield.set_robot_joints(0, action)
        self.playfield.update()
        self.current_state = self.game.compute_state(0)
        reward = self.calculate_reward()
        done = self.check_done()
        info = {}
        return np.array(self.current_state), reward, done, info

    def calculate_reward(self):
        # Define your reward function here
        reward = 0
        return reward

    def check_done(self):
        # Define your termination condition here
        done = False
        return done

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
