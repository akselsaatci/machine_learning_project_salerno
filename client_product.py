from collections import deque
import random
import numpy as np
import sys
import math
import torch
from client import Client, DEFAULT_PORT, JOINTS
from ddpg import DDPG  # Make sure DDPG class is properly imported

TABLE_HEIGHT = 1.0
TABLE_THICKNESS = 0.08
TABLE_LENGTH = 2.4
TABLE_WIDTH = 1.4


class ActionSpace:
    def __init__(self, low, high):
        self.low = np.array(low)
        self.high = np.array(high)
        self.shape = self.low.shape

ACTION_SPACE = ActionSpace(
    [
        -0.3,
        -0.8,
        -10,
        -math.pi / 2,
        -10,
        -math.pi * 3 / 4,
        -10,
        -math.pi * 3 / 4,
        -10,
        -math.pi * 3 / 4,
        -10],
    [
        0.3,
        0.8,
        10,
        math.pi / 2,
        10,
        math.pi * 3 / 4,
        10,
        math.pi * 3 / 4,
        10,
        math.pi * 3 / 4,
        10])


def get_neutral_joint_position():
    jp = [0.0]*JOINTS
    jp[0] = -0.3
    jp[2] = math.pi
    a = math.pi/3.8
    jp[5] = a
    jp[7] = a
    jp[9] = math.pi/3.5
    jp[10] = math.pi/2
    return jp

def run(cli):
    gamma = 0.7
    tau = 0.01
    hidden_size = [256, 256]
    num_inputs = 38
    actor = DDPG(gamma=gamma, tau=tau, hidden_size=hidden_size, num_inputs=num_inputs, action_space=ACTION_SPACE)

    # Initialize Replay Buffer
    buffer_size = 10000  # Maximum size of the replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    max_episodes = 100  # Number of episodes to train
    max_steps = 1000  # Maximum number of steps per episode

    episode_rewards = []  # To store rewards for each episode

    for episode in range(max_episodes):
        episode_reward = 0
        state = cli.get_state()
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert to PyTorch tensor

        for step in range(max_steps):
            # Calculate action
            action = actor.calc_action(state, np.random.normal(0,1,100))

            # Perform action in environment
            cli.send_joints(action.cpu().numpy()[0])

            # Receive next state and reward from environment
            next_state = cli.get_state()
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            reward = get_reward(state[0])  # Example function to get reward from environment
            done = is_done()  # Example function to check if episode is done

            # Store transition in replay buffer
            transition = (state, action, reward, next_state, done)
            replay_buffer.append(transition)

            # Update state
            state = next_state
            episode_reward += reward

            # Perform DDPG updates if replay buffer is sufficiently large
            if len(replay_buffer) > 100:
                batch_size = 32
                batch = random.sample(replay_buffer, batch_size)
                # Perform DDPG updates
                value_loss, policy_loss = actor.update_params(batch)

        # Print episode results
        print(f"Episode {episode + 1}: Reward = {episode_reward}")
        episode_rewards.append(episode_reward)

        # Optionally, save checkpoint
        if (episode + 1) % 10 == 0:
            actor.save_checkpoint(episode + 1, replay_buffer)

    # Optionally, plot episode rewards or perform other analyses
    # ...

    # Finally, save the last checkpoint
    actor.save_checkpoint(max_episodes, replay_buffer)


def get_reward(states):
    ball_x = states[17]
    ball_y = states[19]
    paddle_x = states[11]
    paddle_y = states[13]

    return -((paddle_x - ball_x) ** 2 + (paddle_y - ball_y) ** 2)


def is_done():
    return False



def main():
    name = 'Product Client'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    port = DEFAULT_PORT
    if len(sys.argv) > 2:
        port = sys.argv[2]

    host = 'localhost'
    if len(sys.argv) > 3:
        host = sys.argv[3]

    cli = Client(name, host, port)
    run(cli)



if __name__ == '__main__':
    main()
