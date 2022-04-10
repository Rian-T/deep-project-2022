import os

import retro
import gym
from PIL import Image
from tqdm import tqdm

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback

from wrappers.mario_wrappers import *
from wrappers.retro_wrappers import wrap_deepmind_retro, StochasticFrameSkip

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "SuperMarioKart-Snes",
}

states = ["MarioCircuit_M", "BowserCastle_M", "ChocoIsland_M", "KoopaBeach_M"]

for state in states :
    def make_env():
        env = retro.make(config["env_name"], state)
        env = StochasticFrameSkip(env, n=4, stickprob=0.25)
        env = Discretizer(env, DiscretizerActions.SIMPLE)
        # env= ReduceBinaryActions(env,BinaryActions.SIMPLE)
        env = CutMarioMap(env,show_map=False)
        env = wrap_deepmind_retro(env)    
        # env = RewardScaler(env)
        # env = Monitor(env)  # record stats such as returns
        return env
        

    env = DummyVecEnv([make_env])
    model = PPO.load(f"ppo/{state}/best_model.zip", env)

    if not os.path.isdir(f'generated_dataset/{state}'):
        os.mkdir(f'generated_dataset/{state}')

    print("\nGenerating dataset for state: ", state)

    obs = env.reset()
    for i in tqdm(range(10000)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        action_choosed = action[0]
        dir_path = f'generated_dataset/{state}/{action_choosed}'
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        
        im = Image.fromarray(obs[0, :, :, 0])
        im.save(f"{dir_path}/{i}_{state}_{action_choosed}.png")

        #env.render()
        if done:
            obs = env.reset()
    env.close()