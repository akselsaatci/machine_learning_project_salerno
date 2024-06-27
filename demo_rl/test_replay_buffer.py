import unittest
import torch
import numpy as np
from replay_buffer import ReplayBuffer
import matplotlib.pyplot as plt

class TestReplayBuffer(unittest.TestCase):
    def setUp(self):
        self.state_dim = (3,)
        self.action_dim = (1,)
        self.capacity = 100
        self.buffer = ReplayBuffer(self.capacity)
        self.state = np.zeros(self.state_dim)
        self.action = np.zeros(self.action_dim)
        self.reward = 1.0
        self.done = False
        self.next_state = np.ones(self.state_dim)

    def test_add_and_sample(self):
        print("Adding a sample to the buffer.")
        self.buffer.add(self.state, self.action, self.reward, self.done, self.next_state)

        print(f"Buffer size after adding one sample: {len(self.buffer)}")
        print("Sampling from the buffer.")
        batch = self.buffer.sample(1)
        print(f"Sampled state: {batch.state}")
        print(f"Sampled action: {batch.action}")
        print(f"Sampled reward: {batch.reward}")
        print(f"Sampled done: {batch.done}")
        print(f"Sampled next_state: {batch.next_state}")

        self.assertEqual(batch.state.shape, (1, *self.state_dim))
        self.assertEqual(batch.action.shape, (1, *self.action_dim))
        self.assertEqual(batch.reward.shape, (1, 1))
        self.assertEqual(batch.done.shape, (1, 1))
        self.assertEqual(batch.next_state.shape, (1, *self.state_dim))

        self.assertTrue(torch.equal(batch.state[0], torch.tensor(self.state, dtype=torch.float32)))
        self.assertTrue(torch.equal(batch.action[0], torch.tensor(self.action, dtype=torch.float32)))
        self.assertTrue(torch.equal(batch.reward[0], torch.tensor([self.reward], dtype=torch.float32)))
        self.assertTrue(torch.equal(batch.done[0], torch.tensor([self.done], dtype=torch.float32)))
        self.assertTrue(torch.equal(batch.next_state[0], torch.tensor(self.next_state, dtype=torch.float32)))

    def test_buffer_capacity(self):
        print(f"Adding {self.capacity} samples to the buffer.")
        for _ in range(self.capacity):
            self.buffer.add(self.state, self.action, self.reward, self.done, self.next_state)

        self.assertEqual(len(self.buffer), self.capacity)
        print(f"Buffer size after filling to capacity: {len(self.buffer)}")

        print("Adding one more sample to see if the buffer size remains the same.")
        self.buffer.add(self.state, self.action, self.reward, self.done, self.next_state)
        self.assertEqual(len(self.buffer), self.capacity)
        print(f"Buffer size after adding one more sample: {len(self.buffer)}")

    def test_circular_buffer(self):
        print(f"Adding {self.capacity + 10} samples to the buffer to test circular behavior.")
        for i in range(self.capacity + 10):
            state = np.full(self.state_dim, i, dtype=np.float32)
            self.buffer.add(state, self.action, self.reward, self.done, self.next_state)

        self.assertEqual(len(self.buffer), self.capacity)
        print(f"Buffer size after adding more samples than capacity: {len(self.buffer)}")

        batch = self.buffer.sample(1)
        first_state = batch.state[0].numpy()
        print(f"Sampled state after circular addition: {first_state}")

    def test_sample_batch(self):
        batch_size = 32
        print(f"Adding {self.capacity} random samples to the buffer.")
        for _ in range(self.capacity):
            state = np.random.randn(*self.state_dim).astype(np.float32)
            action = np.random.randn(*self.action_dim).astype(np.float32)
            reward = np.random.rand()
            done = np.random.choice([True, False])
            next_state = np.random.randn(*self.state_dim).astype(np.float32)
            self.buffer.add(state, action, reward, done, next_state)

        print(f"Sampling a batch of {batch_size} from the buffer.")
        batch = self.buffer.sample(batch_size)
        print(f"Sampled batch state shape: {batch.state.shape}")
        print(f"Sampled batch action shape: {batch.action.shape}")
        print(f"Sampled batch reward shape: {batch.reward.shape}")
        print(f"Sampled batch done shape: {batch.done.shape}")
        print(f"Sampled batch next_state shape: {batch.next_state.shape}")

        self.assertEqual(batch.state.shape, (batch_size, *self.state_dim))
        self.assertEqual(batch.action.shape, (batch_size, *self.action_dim))
        self.assertEqual(batch.reward.shape, (batch_size, 1))
        self.assertEqual(batch.done.shape, (batch_size, 1))
        self.assertEqual(batch.next_state.shape, (batch_size, *self.state_dim))

    def visualize_buffer(self):
        print(f"Visualizing the buffer with {self.capacity} samples.")
        for i in range(self.capacity):
            state = np.full(self.state_dim, i, dtype=np.float32)
            self.buffer.add(state, self.action, self.reward, self.done, self.next_state)

        states = np.array([s.numpy().flatten() for s in self.buffer.state])
        plt.figure(figsize=(10, 6))
        plt.plot(states, label='States')
        plt.title('State Values in Replay Buffer')
        plt.xlabel('Sample Index')
        plt.ylabel('State Value')
        plt.legend()
        plt.show()

if __name__ == '__main__':
    unittest.main()
