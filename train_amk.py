import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from amk import RobotTableEnv, STATE_DIMENSION, JOINTS
from collections import deque, namedtuple

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define actor network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        return x

# Define critic network
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Define replay buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states = torch.FloatTensor([e.state for e in batch]).to(device)
        actions = torch.FloatTensor([e.action for e in batch]).to(device)
        rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
        next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
        dones = torch.FloatTensor([e.done for e in batch]).to(device)
        return states, actions, rewards, next_states, dones

# Define DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.001):
        self.actor = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target = ActorNetwork(state_dim, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target = CriticNetwork(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.max_action = max_action
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau

        # Initialize schedulers
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=100, gamma=0.9)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=100, gamma=0.9)

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = self.actor(state).squeeze(0).detach().cpu().numpy()
        action = action + noise_scale * np.random.randn(len(action))
        return np.clip(self.max_action * action, -self.max_action, self.max_action)

    def train(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Ensure states, actions, etc. are on the correct device
        states = states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)
        next_states = next_states.to(device)
        dones = dones.to(device)

        # Update critic
        next_actions = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, next_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_expected, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

        # Step the schedulers
        self.actor_scheduler.step()
        self.critic_scheduler.step()

    def _update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Initialize environment
env = RobotTableEnv()
state_dim = env.observation_space.shape[0]  # Assuming this gives the correct state dimension
action_dim = JOINTS
max_action = 1  # Assuming action space is bounded between -1 and 1
num_episodes = 100
batch_size = 100
start_timesteps = 1000 # Number of initial timesteps to collect data without training


# Initialize agent
agent = DDPGAgent(state_dim, action_dim, max_action)

# Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=100000)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0
    done = False

    while not done:
        if len(replay_buffer.buffer) < start_timesteps:
            action = env.action_space.sample()  # Sample random action during initial exploration
        else:
            action = agent.select_action(state)

        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        if len(replay_buffer.buffer) > batch_size:
            agent.train(replay_buffer, batch_size)

    print(f"Episode: {episode}, Reward: {episode_reward}")
