import sys
import os
import unittest
import numpy as np
import torch
from unittest.mock import patch

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

    @patch.object(DDPG, 'update_params', return_value=(1.0, 1.0))
    def test_train(self, mock_update_params):
        value_loss, policy_loss = self.nn_wrapper.train(self.batch)
        mock_update_params.assert_called_once_with(self.batch)
        self.assertEqual(value_loss, 1.0)
        self.assertEqual(policy_loss, 1.0)

    @patch.object(DDPG, 'save_checkpoint')
    def test_save_checkpoint(self, mock_save_checkpoint):
        self.nn_wrapper.save_checkpoint(self.timestep)
        mock_save_checkpoint.assert_called_once_with(self.timestep, self.nn_wrapper.replay_buffer)

    @patch.object(DDPG, 'load_checkpoint', return_value=True)
    def test_load_checkpoint(self, mock_load_checkpoint):
        result = self.nn_wrapper.load_checkpoint(self.checkpoint_path)
        mock_load_checkpoint.assert_called_once_with(self.checkpoint_path)
        self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
