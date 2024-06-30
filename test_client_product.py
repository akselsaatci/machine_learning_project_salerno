import unittest
from unittest.mock import patch, Mock
from client_product import run, get_neutral_joint_position, calc_action, \
    ACTION_SPACE
import torch
import numpy as np


class TestRunFunction(unittest.TestCase):

    @patch('client_product.Client')
    @patch('ddpg.DDPG')
    def test_run(self, mock_ddpg, mock_client):
        # Mocking the Client instance
        mock_cli = mock_client.return_value
        mock_cli.get_state.return_value = np.random.rand(38)

        # Mocking the DDPG instance
        mock_actor = mock_ddpg.return_value
        mock_actor.calc_action.return_value = torch.FloatTensor(np.random.rand(1, ACTION_SPACE.low.shape[0]))

        # Run the function
        run(mock_cli)

        # Assertions to check if get_state and send_joints were called
        self.assertTrue(mock_cli.get_state.called, "get_state was not called")
        self.assertTrue(mock_cli.send_joints.called, "send_joints was not called")

    def test_get_neutral_joint_position(self):
        # Check if get_neutral_joint_position returns a list of correct length
        neutral_position = get_neutral_joint_position()
        self.assertEqual(len(neutral_position), 11, "Neutral joint position length is incorrect")

    def test_calc_action(self):
        # Test calc_action function
        action = np.random.rand(10)
        y = 0.5
        neutral_position = calc_action(action, y)
        self.assertEqual(len(neutral_position), 11, "Calculated action length is incorrect")
        self.assertAlmostEqual(neutral_position[1], y, "Y position in calculated action is incorrect")


if __name__ == '__main__':
    unittest.main()
