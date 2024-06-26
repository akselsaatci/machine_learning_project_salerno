import unittest
import torch
import gym
import numpy as np
import os
from ddpg import DDPG

class MockEnv:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(action_size,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(state_size,), dtype=np.float32)

    def reset(self):
        return np.zeros(self.state_size, dtype=np.float32)

    def step(self, action):
        next_state = np.random.randn(self.state_size).astype(np.float32)  # Random next state for demonstration
        reward = np.random.rand()  # Random reward for demonstration
        done = np.random.rand() > 0.95  # Random done condition for demonstration
        info = {}
        return next_state, reward, done, info

class ReplayBuffer:
    def __init__(self):
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.next_state = []

    def add(self, state, action, reward, done, next_state):
        self.state.append(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        self.action.append(torch.tensor(action, dtype=torch.float32).unsqueeze(0))
        self.reward.append(torch.tensor([reward], dtype=torch.float32).unsqueeze(0))
        self.done.append(torch.tensor([done], dtype=torch.float32).unsqueeze(0))
        self.next_state.append(torch.tensor(next_state, dtype=torch.float32).unsqueeze(0))

    def sample(self, batch_size):
        idx = np.random.randint(0, len(self.state), size=batch_size)
        batch = type('Batch', (object,), {
            'state': [self.state[i] for i in idx],
            'action': [self.action[i] for i in idx],
            'reward': [self.reward[i] for i in idx],
            'done': [self.done[i] for i in idx],
            'next_state': [self.next_state[i] for i in idx]
        })()
        return batch

class TestDDPG(unittest.TestCase):

    def setUp(self):
        self.state_size = 3
        self.action_size = 1
        self.env = MockEnv(self.state_size, self.action_size)
        self.ddpg = DDPG(
            gamma=0.99,
            tau=0.001,
            hidden_size=[400, 300],
            num_inputs=self.state_size,
            action_space=self.env.action_space
        )

    def test_initialization(self):
        self.assertIsNotNone(self.ddpg.actor)
        self.assertIsNotNone(self.ddpg.critic)
        self.assertIsNotNone(self.ddpg.actor_target)
        self.assertIsNotNone(self.ddpg.critic_target)

    def test_calc_action(self):
        state = torch.tensor(self.env.reset(), dtype=torch.float32).unsqueeze(0)
        action = self.ddpg.calc_action(state)
        self.assertEqual(action.shape, (1, self.action_size))

    def test_update_params(self):
        replay_buffer = ReplayBuffer()
        for _ in range(10):
            state = self.env.reset()
            action = self.env.action_space.sample()
            next_state, reward, done, _ = self.env.step(action)
            replay_buffer.add(state, action, reward, done, next_state)

        batch = replay_buffer.sample(10)
        value_loss, policy_loss = self.ddpg.update_params(batch)
        self.assertIsInstance(value_loss, float)
        self.assertIsInstance(policy_loss, float)

    def test_checkpoint_save_load(self):
        replay_buffer = ReplayBuffer()
        checkpoint_path = self.ddpg.checkpoint_dir + '/ep_0.pth.tar'
        os.makedirs(self.ddpg.checkpoint_dir, exist_ok=True)  # Ensure directory exists
        self.ddpg.save_checkpoint(0, replay_buffer)
        self.assertTrue(os.path.isfile(checkpoint_path))

        start_timestep, loaded_replay_buffer = self.ddpg.load_checkpoint(checkpoint_path)
        self.assertEqual(start_timestep, 1)
        self.assertEqual(len(loaded_replay_buffer.state), 0)

    def test_learning(self):
        replay_buffer = ReplayBuffer()
        num_episodes = 10
        max_steps = 1000
        batch_size = 10

        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(max_steps):
                action = self.ddpg.calc_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
                next_state, reward, done, _ = self.env.step(action)
                replay_buffer.add(state, action, reward, done, next_state)
                state = next_state
                episode_reward += reward

                if len(replay_buffer.state) > batch_size:
                    batch = replay_buffer.sample(batch_size)
                    value_loss, policy_loss = self.ddpg.update_params(batch)
                    print(f'Episode: {episode}, Step: {step}, Value Loss: {value_loss:.4f}, Policy Loss: {policy_loss:.4f}')

                if done:
                    break

            print(f'Episode: {episode}, Total Reward: {episode_reward:.4f}')

if __name__ == '__main__':
    unittest.main()
