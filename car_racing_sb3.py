import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

def make_env():
    env = gym.make("CarRacing-v2", render_mode="rgb_array", continuous=True)
    env = Monitor(env)
    return env

env = DummyVecEnv([make_env])

model = PPO(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log="./car_racing_tensorboard/"
)

model.learn(total_timesteps=100_000, tb_log_name="PPO_CarRacing")
model.save("/home/ubuntu/persistent/ppo_car_racing")
env.close()
print("Training complete!")
