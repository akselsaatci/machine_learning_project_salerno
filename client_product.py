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
    #Used Joints: Everything but joint[1]
    #Actionspace: 0,2,3,...,11
    [-0.1, -0.001, -0.7, -0.005, -0.7, -0.01, -0.7, -0.05, -1.4, -0.1],
    [0.5, 0.001, 1.5, 0.005, 0.7, 0.01, 0.7, 0.05, 0.7, 0.1])


def get_neutral_joint_position():
    jp = [0.0]*JOINTS
    jp[0] = -0.2
    jp[2] = math.pi
    a = math.pi/3.8
    jp[3] = 0.1
    jp[5] = a
    jp[7] = a
    jp[9] = math.pi/3.5
    jp[10] = math.pi/2
    return jp


class OrnsteinUhlenbeckNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
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




gamma = 0.7
tau = 0.1
hidden_size = [256, 256]
num_inputs = 38
#Initialize Actor
actor = DDPG(gamma=gamma, tau=tau, hidden_size=hidden_size, num_inputs=num_inputs, action_space=ACTION_SPACE)
actor.load_checkpoint()
def run(cli):


    # Initialize Replay Buffer
    buffer_size = 10000  # Maximum size of the replay buffer
    replay_buffer = deque(maxlen=buffer_size)

    max_episodes = 5000  # Number of episodes to train
    max_steps = 400  # Maximum number of steps per episode
    episode_rewards = []  # To store rewards for each episode

    # Assume action_space is an instance of ActionSpace with appropriate low and high bounds
    action_dim = ACTION_SPACE.low.shape[0]
    noise = OrnsteinUhlenbeckNoise(size=action_dim)
    for episode in range(max_episodes):
        episode_reward = 0
        state = cli.get_state()
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert to PyTorch tensor
        rob_touch_switch= False #switch for the robo touch (Workaround bc of engine loading)

        for step in range(max_steps):
            reward=0
            noise.reset()
            # Calculate action
            action = actor.calc_action(state,noise)
            act_action = action.cpu().numpy()[0]
            '''#Visualization of ACTION_SPACE
            if episode%6 <3:
                act_action[1] = ACTION_SPACE.low[1]
                act_action[3] = ACTION_SPACE.low[3]
                act_action[5] = ACTION_SPACE.low[5]
                act_action[7] = ACTION_SPACE.low[7]
            else:
                act_action[1] = ACTION_SPACE.high[1]
                act_action[3] = ACTION_SPACE.high[3]
                act_action[5] = ACTION_SPACE.high[5]
                act_action[7] = ACTION_SPACE.high[7]'''
            act_action = calc_action(act_action, state[0][17])
            # Perform action in environment
            cli.send_joints(act_action)

            # Receive next state and reward from environment
            next_state = cli.get_state()
            next_state = torch.FloatTensor(next_state).unsqueeze(0)

            #via a switch variable we give a reward, if the robot touches the ball
            if not rob_touch_switch and not state[0][31]:
                rob_touch_switch = True
            if state[0][31] and rob_touch_switch:
                reward = 18
            reward = reward + get_reward(state[0])  #  function to get reward from environment
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
        if (episode + 1) % 20 == 0:
            actor.save_checkpoint(episode + 1, replay_buffer)

    # Optionally, plot episode rewards or perform other analyses
    # ...

    # Finally, save the last checkpoint
    actor.save_checkpoint(max_episodes, replay_buffer)


def get_reward(states):
    reward_versor=(states[14]-0.02) **2 + (states[15]-0.90) ** 2 + (states[16]-0.43) ** 2
    #print(f"Z: {reward_z}, Versor: {0.5*reward_versor}")

    return -(8 + 1 * reward_versor)

def calc_action(action, y): #uses the standard position and adds changes
    a= get_neutral_joint_position()
    #print(f"Actions: {action}")
    a[1]= y
    a[0]= a[0] + action[0]
    for i in range(len(action)-1):
        a[i+2]= a[i+2] + action[i+1]
    return a

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
