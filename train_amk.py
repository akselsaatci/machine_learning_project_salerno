import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

from joblib import Parallel, delayed

from amk import RobotTableEnv, STATE_DIMENSION, JOINTS
from collections import deque, namedtuple

# Define actor network
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, controllable_action_dim):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, controllable_action_dim)

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

    def sample(self, batch_size):#Takes samples of old training data ! needs to be optimized
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*[(e.state, e.action, e.reward, e.next_state, e.done) for e in batch])


        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        return states, actions, rewards, next_states, dones

# Define DDPG agent
class DDPGAgent:
    def __init__(self, state_dim, total_action_dim, controllable_action_dim, max_action, actor_lr=0.001, critic_lr=0.001, gamma=0.99, tau=0.001):
        self.actor = ActorNetwork(state_dim, controllable_action_dim)
        self.actor_target = ActorNetwork(state_dim, controllable_action_dim)
        self.critic = CriticNetwork(state_dim, total_action_dim)
        self.critic_target = CriticNetwork(state_dim, total_action_dim)
        self.max_action = max_action
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.gamma = gamma
        self.tau = tau
        self.controllable_action_dim = controllable_action_dim
        self.total_action_dim = total_action_dim
        self.movable_indices = [0, 1, 3, 5, 7, 9]

        # Copy the weights from actor to target_actor and critic to target_critic
        self._hard_update(self.actor_target, self.actor)

    def select_action(self, state, current_action, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        controllable_action = self.actor(state).squeeze(0).detach().numpy()
        controllable_action = np.clip(controllable_action, -1, 1)
        controllable_action = self.max_action * controllable_action

        # Add exploration noise
        controllable_action += noise_scale * np.random.randn(self.controllable_action_dim)
        controllable_action = np.clip(controllable_action, -self.max_action, self.max_action)

        # Merge controllable action with current action
        new_action = np.copy(current_action)
        for idx, action_value in zip(self.movable_indices, controllable_action):
            new_action[idx] = action_value
        return new_action

    def train(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # Normalize rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Update critic
        next_actions = self.actor_target(next_states)
        q_targets_next = self.critic_target(next_states, next_actions)
        q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))
        q_expected = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_expected, q_targets.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        full_actions = torch.zeros((batch_size, self.total_action_dim))
        for i in range(batch_size):
            for idx, action_value in zip(self.movable_indices, predicted_actions[i]):
                full_actions[i][idx] = action_value

        actor_loss = -self.critic(states, full_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self._update_target_networks()

    def _update_target_networks(self):
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data)

    def _hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

# Initialize environment
env = RobotTableEnv()
# Assuming state_dim matches observation space size
state_dim = env.observation_space.shape[0]
total_action_dim = JOINTS
controllable_action_dim = len([0, 1, 3, 5, 7, 9])
max_action = 10  # Assuming action space is bounded between -1 and 1
#play with the following three to get better trainings
num_episodes = 200 #length of training
batch_size = 512 # number of old data, that gets input -> rn huge factor on performance
max_steps_per_episode = 100  # Maximum number of steps per episode

# Initialize agent
agent = DDPGAgent(state_dim, total_action_dim, controllable_action_dim, max_action)

# Initialize replay buffer
replay_buffer = ReplayBuffer(capacity=100000)

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    current_action = np.zeros(total_action_dim)  # Initialize current action for all joints
    current_action[10] = 5  # Truns the paddle
    episode_reward = 0
    done = False
    step_count = 0  # Initialize step counter

    while not done and step_count < max_steps_per_episode:
        #time.sleep(0.0005)
        action = agent.select_action(state, current_action)
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, done)
        episode_reward += reward
        state = next_state

        #Paddle good? stop turning
        if state[10]%3.14<0.1 or state[10]%3.14>3.04: current_action[10]=0
        current_action = action  # Update current action
        step_count += 1  # Increment step counter
        #print(f"Episode: {episode}, Step: {step_count}, Reward: {episode_reward}")
        #print(f"Paddle-Turn ={state[10]}")
        if len(replay_buffer.buffer) > batch_size:
            agent.train(replay_buffer, batch_size)

    print(f"Episode: {episode}, Reward: {episode_reward}")
