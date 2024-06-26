import gym
import numpy as np
import torch
from ddpg import DDPG
from replay_buffer import ReplayBuffer

# Create the environment (replace with your table tennis environment)
env = gym.make('TableTennisTrainingEnv-v0')

# Initialize the DDPG agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
ddpg_agent = DDPG(
    gamma=0.99,
    tau=0.001,
    hidden_size=[400, 300],
    num_inputs=state_size,
    action_space=env.action_space
)

# Initialize replay buffer
replay_buffer = ReplayBuffer(100)

# Training parameters
num_episodes = 1000
max_steps = 1000
batch_size = 64

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = ddpg_agent.calc_action(torch.tensor(state, dtype=torch.float32).unsqueeze(0)).cpu().numpy()[0]
        next_state, reward, done, _ = env.step(action)
        replay_buffer.add(state, action, reward, done, next_state)
        state = next_state
        episode_reward += reward

        if len(replay_buffer.state) > batch_size:
            batch = replay_buffer.sample(batch_size)
            value_loss, policy_loss = ddpg_agent.update_params(batch)
            print(f'Episode: {episode}, Step: {step}, Value Loss: {value_loss:.4f}, Policy Loss: {policy_loss:.4f}')

        if done:
            break

    print(f'Episode: {episode}, Total Reward: {episode_reward:.4f}')
