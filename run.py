import gymnasium as gym
import gym_carla
import carla
from stable_baselines3 import SAC

def main():
  params = {
    'number_of_vehicles': 1,
    'number_of_walkers': 0,
    'display_size': 256,
    'max_past_step': 1,
    'dt': 0.1,
    'discrete': False,
    'discrete_acc': [-3.0, 0.0, 3.0],
    'discrete_steer': [-0.2, 0.0, 0.2],
    'continuous_accel_range': [-3.0, 3.0],
    'continuous_steer_range': [-0.3, 0.3],
    'ego_vehicle_filter': 'vehicle.lincoln*',
    'port': 4000,
    'town': 'Town03',
    'max_time_episode': 1000,
    'max_waypt': 12,
    'obs_range': 32,
    'lidar_bin': 0.125,
    'd_behind': 12,
    'out_lane_thres': 2.0,
    'desired_speed': 8,
    'max_ego_spawn_times': 200,
    'display_route': False,
  }
  env = gym.make('carla-v0', params=params)
  model = SAC("MlpPolicy", env, device="cuda", buffer_size=20000, verbose=1, tensorboard_log="./tensorboard_DQN/")
  obs, info = env.reset()
  
  while True:
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

if __name__ == '__main__':
  main()
