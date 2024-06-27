import gym
import numpy as np
import torch
from ddpg import DDPG
from replay_buffer import ReplayBuffer

state_size = 5
action_size = 5


class NNWrapper:
    def __init__(self):
        self.ddpg = DDPG(
            gamma=0.99,
            tau=0.001,
            hidden_size=[400, 300],  # Adjust based on your specific network architecture
            num_inputs=state_size,  # Define state_size according to your state representation
            action_space=gym.spaces.Box(low=-1, high=1, shape=(action_size,), dtype=np.float32)
        )
        self.replay_buffer = ReplayBuffer(capacity=1000000)  # Example capacity, adjust as needed

    def update(self, state):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
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
