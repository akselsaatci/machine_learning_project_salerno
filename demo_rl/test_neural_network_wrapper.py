import sys
import os
import unittest
import numpy as np
import torch
from unittest.mock import MagicMock

# Add the directory containing 'demo_rl' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_network_wrapper import NNWrapper
from replay_buffer import ReplayBuffer
from ddpg import DDPG


class TestNNWrapper(unittest.TestCase):

    def setUp(self):
        self.nn_wrapper = NNWrapper()
        self.state = np.random.randn(5)
        self.batch = {
            'state': torch.randn(32, 5),
            'action': torch.randn(32, 5),
            'reward': torch.randn(32, 1),
            'next_state': torch.randn(32, 5),
            'done': torch.randint(0, 2, (32, 1)).float()
        }
        self.timestep = 1000
        self.checkpoint_path = 'dummy_path'

    def test_initialization(self):
        self.assertIsInstance(self.nn_wrapper.ddpg, DDPG)
        self.assertIsInstance(self.nn_wrapper.replay_buffer, ReplayBuffer)

    def test_update(self):
        action = self.nn_wrapper.update(self.state)
        self.assertEqual(action.shape, (5,))
        self.assertTrue(np.all(action >= -1) and np.all(action <= 1))

    def test_train(self):
        # Mocking the DDPG update_params method
        self.nn_wrapper.ddpg.update_params = MagicMock(return_value=(1.0, 1.0))
        value_loss, policy_loss = self.nn_wrapper.train(self.batch)
        self.nn_wrapper.ddpg.update_params.assert_called_once_with(self.batch)
        self.assertEqual(value_loss, 1.0)
        self.assertEqual(policy_loss, 1.0)

    def test_save_checkpoint(self):
        # Mocking the DDPG save_checkpoint method
        self.nn_wrapper.ddpg.save_checkpoint = MagicMock()
        self.nn_wrapper.save_checkpoint(self.timestep)
        self.nn_wrapper.ddpg.save_checkpoint.assert_called_once_with(self.timestep, self.nn_wrapper.replay_buffer)

    def test_load_checkpoint(self):
        # Mocking the DDPG load_checkpoint method
        self.nn_wrapper.ddpg.load_checkpoint = MagicMock(return_value=True)
        result = self.nn_wrapper.load_checkpoint(self.checkpoint_path)
        self.nn_wrapper.ddpg.load_checkpoint.assert_called_once_with(self.checkpoint_path)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
