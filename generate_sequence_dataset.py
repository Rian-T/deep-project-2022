import os
from statistics import mean

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
    frames = []
    count = 0
    action_buffer = []

    obs = env.reset()
    for i in tqdm(range(20000)):
        action, _state = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        action_buffer.append(action[0])
        
        # Concatenate last 8 frames
        frames.append(obs[0, :, :, 0])

        if len(frames) == 8:
            # last 4 frames
            eligible_actions = action_buffer[-4:]
            action_choosed = max(eligible_actions,key=eligible_actions.count)
            dir_path = f'generated_dataset/{state}/{action_choosed}'
            if not os.path.isdir(dir_path):
                os.mkdir(dir_path)
            os.mkdir(dir_path+f"/{count}")
            for j in range(8):
                im = Image.fromarray(frames[j])
                im.save(f"{dir_path}/{count}/{j}_{state}_{action_buffer[j]}.png")
            count += 1
            frames = []
            action_buffer = []

        #env.render()
        if done:
            obs = env.reset()
            frames = []
            action_buffer = []
    env.close()