import gym
from testo import TableTennisEnv
from testo import JOINTS, STATE_DIMENSION
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from aksel import NormalGame

game = NormalGame()
env = TableTennisEnv(game)

# Check if the environment follows the Gym API
check_env(env)

# Create the RL agent
model = PPO("MlpPolicy", env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_table_tennis")

# Evaluate the agent
obs = env.reset()
for _ in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()