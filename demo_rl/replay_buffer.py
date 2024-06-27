import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.position = 0
        self.size = 0

        self.state = torch.zeros((capacity, *state_dim), dtype=torch.float32)
        self.action = torch.zeros((capacity, *action_dim), dtype=torch.float32)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32)
        self.done = torch.zeros((capacity, 1), dtype=torch.float32)
        self.next_state = torch.zeros((capacity, *state_dim), dtype=torch.float32)

    def add(self, state, action, reward, done, next_state):
        self.state[self.position] = torch.tensor(state, dtype=torch.float32)
        self.action[self.position] = torch.tensor(action, dtype=torch.float32)
        self.reward[self.position] = torch.tensor([reward], dtype=torch.float32)
        self.done[self.position] = torch.tensor([done], dtype=torch.float32)
        self.next_state[self.position] = torch.tensor(next_state, dtype=torch.float32)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        batch = type('Batch', (object,), {
            'state': self.state[idx],
            'action': self.action[idx],
            'reward': self.reward[idx],
            'done': self.done[idx],
            'next_state': self.next_state[idx]
        })()
        return batch

    def __len__(self):
        return self.size


# Example usage
state_dim = (38,)  # Replace with actual state dimension
action_dim = (11,)  # Replace with actual action dimension
buffer = ReplayBuffer(capacity=10000)

# Add samples
#buffer.add([0, 1, 2], [0], 1.0, False, [0, 1, 3])

# Sample a batch
batch = buffer.sample(32)
print(batch.state.shape)  # Should print torch.Size([32, 3])
