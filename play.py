import os

import retro
import gym
from PIL import Image

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

#states = ["MarioCircuit_M", "BowserCastle_M", "DonutPlains_M", "GhostValley_M", "ChocoIsland_M", "KoopaBeach_M", "RainbowRoad_M"]
state = "MarioCircuit_M"

def make_env():
    env = retro.make(config["env_name"], state)
    # env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    # env= ReduceBinaryActions(env,BinaryActions.SIMPLE)
    env = CutMarioMap(env,show_map=False)
    env = wrap_deepmind_retro(env)    
    # env = RewardScaler(env)
    # env = Monitor(env)  # record stats such as returns
    return env
    

env = DummyVecEnv([make_env])
model = PPO.load(f"ppo/{state}/best_model.zip", env)


obs = env.reset()
for i in range(100000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)

    env.render()
    if done:
      obs = env.reset()