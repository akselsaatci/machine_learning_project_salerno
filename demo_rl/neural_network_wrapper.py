import numpy as np
import torch
from ddpg import DDPG
from gym import spaces
from replay_buffer import ReplayBuffer


class NNWrapper:
    def __init__(self):
        self.ddpg = DDPG(gamma=0.99, tau=0.001, hidden_size=[400, 300], num_inputs=37,
                         action_space= spaces.Box(
            low=np.array([-0.3, -0.8, -np.inf, -np.pi/2, -np.inf, -3*np.pi /
                         4, -np.inf, -3*np.pi/4, -np.inf, -3*np.pi/4, -np.inf, -np.inf]),
            high=np.array([0.3, 0.8, np.inf, np.pi/2, np.inf, 3*np.pi/4,
                           np.inf, 3*np.pi/4, np.inf, 3*np.pi/4, np.inf, np.inf]),
            dtype=np.float32))
        self.replay_buffer = ReplayBuffer(capacity=1000000)  # Example capacity, adjust as needed

    def update(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.ddpg.calc_action(state_tensor)
        action_np = action.squeeze(0).cpu().numpy()
        return action_np

    def train(self, batch):
        value_loss, policy_loss = self.ddpg.update_params(batch)
        return value_loss, policy_loss

    def save_checkpoint(self, timestep):
        self.ddpg.save_checkpoint(timestep, self.replay_buffer)

    def load_checkpoint(self, checkpoint_path=None):
        return self.ddpg.load_checkpoint(checkpoint_path)


nn_wrapper = NNWrapper()
