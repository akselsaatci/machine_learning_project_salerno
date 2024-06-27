import sys
import os
import unittest
from unittest.mock import MagicMock, patch
import numpy as np

# Add the directory containing 'demo_rl' and 'channel' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from demo_rl.client import Client, JOINTS, STATE_DIMENSION, DEFAULT_PORT

class TestClient(unittest.TestCase):

    @patch('demo_rl.client.channel.ClientChannel')
    def setUp(self, MockClientChannel):
        self.mock_channel = MockClientChannel.return_value
        self.client = Client(name='TestClient')

    @patch('demo_rl.client.channel.decode_float_list', return_value=[0.1] * STATE_DIMENSION)
    def test_get_state(self, mock_decode_float_list):
        self.mock_channel.receive.side_effect = [b'some_data', None]
        state = self.client.get_state()
        self.assertIsNotNone(state)
        self.assertIsInstance(state, np.ndarray)
        self.assertEqual(state.shape, (STATE_DIMENSION,))
        self.mock_channel.receive.assert_called()
        mock_decode_float_list.assert_called()

    @patch('demo_rl.client.channel.encode_float_list', return_value=b'encoded_data')
    def test_send_joints(self, mock_encode_float_list):
        joints = [0.1] * JOINTS
        self.client.channel.send = MagicMock()
        self.client.send_joints(joints)
        self.client.channel.send.assert_called_once_with(b'encoded_data')
        mock_encode_float_list.assert_called_once_with(joints)

    @patch('demo_rl.client.channel.encode_float_list', return_value=None)
    def test_send_joints_invalid_vector(self, mock_encode_float_list):
        joints = [0.1] * JOINTS
        with self.assertRaises(ValueError):
            self.client.send_joints(joints)

    def test_send_joints_invalid_length(self):
        joints = [0.1] * (JOINTS - 1)
        with self.assertRaises(ValueError):
            self.client.send_joints(joints)

    @patch('demo_rl.client.NNWrapper')
    @patch.object(Client, 'get_state', side_effect=[np.random.randn(STATE_DIMENSION), None])
    @patch.object(Client, 'send_joints')
    def test_run(self, mock_send_joints, mock_get_state, MockNNWrapper):
        mock_nn_wrapper = MockNNWrapper.return_value
        mock_nn_wrapper.update.return_value = [0.1] * JOINTS

        self.client.run()

        self.assertEqual(mock_get_state.call_count, 2)
        mock_nn_wrapper.update.assert_called_once()
        mock_send_joints.assert_called_once_with([0.1] * JOINTS)

    def test_close(self):
        self.client.channel.close = MagicMock()
        self.client.close()
        self.client.channel.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
