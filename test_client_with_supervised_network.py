import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# Mock implementation of BallToJointsNN
class BallToJointsNN(nn.Module):
    def __init__(self, input_size=3, output_size=11):
        super(BallToJointsNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x


def mock_load_model():
    return BallToJointsNN()


class TestBallToJointsNN(unittest.TestCase):
    def setUp(self):
        self.model = BallToJointsNN()
        self.input_data = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        self.expected_output = torch.tensor([[0.1] * 11], dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            self.predicted_output = self.model(self.input_data)

    def test_model_forward(self):
        self.assertEqual(self.predicted_output.shape, (1, 11))

    def test_model_output(self):
        self.assertFalse(torch.allclose(self.predicted_output, self.expected_output, atol=1e-2))


class TestRunFunction(unittest.TestCase):
    @patch('client_with_supervised_network.Client')
    @patch('client_with_supervised_network.pickle.load', side_effect=mock_load_model)
    @patch('client_with_supervised_network.StandardScaler')
    @patch('client_with_supervised_network.train_test_split')
    def test_run(self, mock_train_test_split, mock_standard_scaler, mock_pickle_load, mock_client):
        mock_client_instance = MagicMock()
        mock_client.return_value = mock_client_instance

        mock_scaler_X = MagicMock()
        mock_scaler_y = MagicMock()
        mock_standard_scaler.side_effect = [mock_scaler_X, mock_scaler_y]

        # Mock scaler to handle 2D input
        mock_scaler_X.transform.side_effect = lambda x: np.array([[0.0, 0.0, 0.0]])
        mock_scaler_y.inverse_transform.return_value = np.array([0.0] * 11)

        mock_train_test_split.return_value = (torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
                                              torch.tensor([[0.1] * 11], dtype=torch.float32),
                                              torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
                                              torch.tensor([[0.1] * 11], dtype=torch.float32))

        # Import the 'run' function inside the test to ensure mocking works
        from client_with_supervised_network import run
        run(mock_client_instance)

        # Verify the client methods are called
        mock_client_instance.get_state.assert_called()
        mock_client_instance.send_joints.assert_called_with([0.0] * 11)


if __name__ == '__main__':
    unittest.main()

