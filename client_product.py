from collections import deque
import random
import numpy as np
import sys
import math
import torch
from client import Client, DEFAULT_PORT
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
    [-0.3, -0.8, -10, -math.pi / 2, -10, -math.pi * 3 / 4, -10, -math.pi * 3 / 4, -10, -math.pi * 3 / 4, -10],
    [0.3, 0.8, 10, math.pi / 2, 10, math.pi * 3 / 4, 10, math.pi * 3 / 4, 10, math.pi * 3 / 4, 10])




class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.4):
        self.size = size
        self.mu = mu #mean of noise. Initially 0.0
        self.theta = theta #decline of exploration. Initially 0.15
        self.sigma = sigma #exploraion. Initialy 0.2
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)
        self.state = x + dx
        return self.state





def run(cli):
    gamma = 0.7
    tau = 0.1
    hidden_size = [256, 256]
    num_inputs = 38
    #Initialize Actor
    actor = DDPG(gamma=gamma, tau=tau, hidden_size=hidden_size, num_inputs=num_inputs, action_space=ACTION_SPACE)
    actor.load_checkpoint()

    # Initialize Replay Buffer
    buffer_size = 10000  # Maximum size of the replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    max_episodes = 1000  # Number of episodes to train
    max_steps = 400  # Maximum number of steps per episode

    episode_rewards = []  # To store rewards for each episode

    for episode in range(max_episodes):
        episode_reward = 0
        state = cli.get_state()
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert to PyTorch tensor

        # Assume action_space is an instance of ActionSpace with appropriate low and high bounds
        action_dim = ACTION_SPACE.low.shape[0]
        noise = OrnsteinUhlenbeckNoise(size=action_dim)
        for step in range(max_steps):


            # Calculate action
            action = actor.calc_action(state,noise)

            # Perform action in environment
            cli.send_joints(action.cpu().numpy()[0])

            # Receive next state and reward from environment
            next_state = cli.get_state()
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            reward = get_reward(state[0])  # Example function to get reward from environment
            done = is_done()  # Example function to check if episode is done

            # Store transition in replay buffer
            transition = (state, action,reward, next_state,done)
            replay_buffer.append(transition)

            #end episode, if point scored
            if state[0][35] < next_state[0][35] or state[0][34] < next_state[0][34]:
                break

            # Update state
            state = next_state
            episode_reward += reward

            # Perform DDPG updates if replay buffer is sufficiently large
            if len(replay_buffer) > 100:
                #print(f"Training...")
                batch_size = 32
                batch = random.sample(replay_buffer, batch_size)
                # Perform DDPG updates
                value_loss, policy_loss = actor.update_params(batch)

        #here im thinking about clearing the buffer, so only new values are in there
        #replay_buffer.clear()

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
    reward_x= (states[11] - states[17]) ** 2
    reward_z= (states[13] - states[19]) ** 2
    reward_versor=(states[14]-0.02) **2 + (states[15]-0.90) ** 2 + (states[16]-0.43) ** 2
    reward_stance= (states[2]-math.pi)**2 + (states[5]-math.pi/3.8) ** 2 + (states[7]-math.pi/3.8) ** 2 + (states[9]-math.pi/3.5) ** 2
    #print(f"X: {0.1*reward_x}, Z: {0.1*reward_z}, Versor: {0.5*reward_versor}, Stance: {0.5*reward_stance}")
    return -(0.1*reward_x + 0.1* reward_z + 0.5*reward_versor+ 0.5*reward_stance)


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
