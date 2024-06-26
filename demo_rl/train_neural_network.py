import argparse
import logging
import os

import gym
import numpy as np
import torch

from ddpg import DDPG
from wrappers import NormalizedActions


def initialize_agent():

    # Here should be point where we initialize NN, initial variables, etc.

    return


def initialize_environment():

    # Here should be point where we start the "server" (fake_server) - initialize game

    return


# Create logger
logger = logging.getLogger('test')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

# Libdom raises an error if this is not set to true on Mac OSX
# see https://github.com/openai/spinningup/issues/16 for more information
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Parse given arguments
parser = argparse.ArgumentParser()
parser.add_argument("--env", default="RoboschoolInvertedPendulumSwingup-v1",
                    help="Env. on which the agent should be trained")
parser.add_argument("--render", default="True", help="Render the steps")
parser.add_argument("--seed", default=0, help="Random seed")
parser.add_argument("--save_dir", default="./saved_models/", help="Dir. path to load a model")
parser.add_argument("--episodes", default=100, help="Num. of test episodes")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Some parameters, which are not saved in the network files
gamma = 0.99  # discount factor for reward (default: 0.99)
tau = 0.001  # discount factor for model (default: 0.001)
hidden_size = (400, 300)  # size of the hidden layers (Deepmind: 400 and 300; OpenAI: 64)

max_episodes = args.episodes

max_steps_per_episode = 1000 # temp default value

if __name__ == "__main__":

    logger.info("Using device: {}".format(device))

    # Create the env
    kwargs = dict()
    if args.env == 'TableTennisLearning-v1':
        kwargs['swingup'] = True
    env = gym.make(args.env, **kwargs)
    env = NormalizedActions(env)

    # Setting rnd seed for reproducibility
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = args.save_dir + args.env

    agent = DDPG(gamma,
                 tau,
                 hidden_size,
                 env.observation_space.shape[0],
                 env.action_space,
                 checkpoint_dir=checkpoint_dir
                 )

    agent.load_checkpoint()

    # Load the agents parameters
    agent.set_eval()

    initialize_environment()
    initialize_agent()

    for episode in range(max_episodes):
        step = 0
        returns = list()
        state = torch.Tensor([env.reset()]).to(device)
        episode_reward = 0

        for step in range(max_steps_per_episode):
            action = agent.calc_action(state, action_noise=None)
            next_state, reward, done, episode = env.step(action.cpu().numpy()[0])
            q_value = agent.critic(state, action)

            #agent.store_experience(state, action, reward, next_state, done)
            #agent.update_policy()

            next_state, reward, done, episode = env.step(action.cpu().numpy()[0])
            episode_reward += reward
            state = torch.Tensor([next_state]).to(device)

            agent.save_checkpoint(state, action, reward, next_state, done)
            agent.update_policy()

            step += 1

            if done:
                logger.info(episode_reward)
                returns.append(episode_reward)
                break

    mean = np.mean(returns)
    variance = np.var(returns)
    logger.info("Score (on 100 episodes): {} +/- {}".format(mean, variance))


    # log_episode_reward(episode, episode_reward)
    #
    # if episode % evaluation_interval == 0:
    #     evaluate_agent_performance()
    #
    # agent.decay_exploration_rate()
    #
    # if early_stopping_criteria_met():
    #     break