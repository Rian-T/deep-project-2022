import retro
from PIL import Image
import time

from gym.wrappers import FrameStack

import torch
import torch.nn as nn
from torchvision import transforms

from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

from wrappers.mario_wrappers import *
from wrappers.retro_wrappers import wrap_deepmind_retro, StochasticFrameSkip

from model.basic import CNNModel, MLPModel, ResNetModel, CnnLSTMModel

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "SuperMarioKart-Snes",
}

#states = ["MarioCircuit_M", "BowserCastle_M", "DonutPlains_M", "GhostValley_M", "ChocoIsland_M", "KoopaBeach_M", "RainbowRoad_M"]
state = "MarioCircuit_M"

def make_env():
    env = retro.make(config["env_name"], state)
    # env = StochasticFrameSkip(env, n=8, stickprob=0.25)
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    # env= ReduceBinaryActions(env,BinaryActions.SIMPLE)
    env = CutMarioMap(env,show_map=False)
    env = wrap_deepmind_retro(env)
    env = FrameStack(env, 16)
    # env = RewardScaler(env)
    # env = Monitor(env)  # record stats such as returns
    return env
    

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/CNNLSTM/{state}", record_video_trigger=lambda x: x % 2000 == 0, video_length=200)
#model = CNNModel.load_from_checkpoint(f"checkpoints/CNN/deep-project-2022/2a5qk2g1/checkpoints/epoch=10-step=1166.ckpt")
#model = MLPModel.load_from_checkpoint(f"checkpoints/MLP/lightning_logs/version_3/checkpoints/epoch=61-step=6572.ckpt")
#model = ResNetModel.load_from_checkpoint(f"checkpoints/RESNET/deep-project-2022/1ivbxl2j/checkpoints/epoch=6-step=2954.ckpt")
model = CnnLSTMModel.load_from_checkpoint(f"checkpoints/LSTM/deep-project-2022/3gmijjvg/checkpoints/epoch=20-step=2688.ckpt")

#model.eval()

norm = transforms.Normalize(0.4505, 0.1786)

obs = env.reset()
obs = torch.from_numpy(obs[:, :, :, :, 0]).float()
obs = norm(obs)

for i in range(10000):
    with torch.no_grad():
        action = model(obs)
    print(action)
    action = torch.argmax(action, dim=1).cpu().detach().numpy()
    print(action)
    obs, reward, done, info = env.step(action)
    
    obs = torch.from_numpy(obs[:, :, :, :, 0]).float()
    obs = norm(obs)

    #time.sleep(0.02)
    env.render()
    #time.sleep(1)
    if done:
        obs = env.reset()
        obs = torch.from_numpy(obs[:, :, :, :, 0]).float()
        obs = norm(obs)
