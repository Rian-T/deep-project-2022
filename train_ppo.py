import retro

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder

import wandb
from wandb.integration.sb3 import WandbCallback

from wrappers.mario_wrappers import *
from wrappers.retro_wrappers import wrap_deepmind_retro, StochasticFrameSkip
from wrappers.utils import SaveOnBestTrainingRewardCallbackCustom

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "SuperMarioKart-Snes",
}

wandb.tensorboard.patch(root_logdir="./runs")
run = wandb.init(
    project="deep-project-2022",
    config=config,
    #sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

states = ["MarioCircuit_M", "BowserCastle_M", "DonutPlains_M", "GhostValley_M", "ChocoIsland_M", "KoopaBeach_M", "RainbowRoad_M"]
state = "MarioCircuit_M"

def make_env():
    env = retro.make(config["env_name"], state)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    env = Discretizer(env, DiscretizerActions.SIMPLE)
    # env= ReduceBinaryActions(env,BinaryActions.SIMPLE)
    env = TimeLimitWrapperMarioKart(env, minutes=3,seconds=0)
    env = CutMarioMap(env, show_map=False)
    env = wrap_deepmind_retro(env)    
    # env = RewardScaler(env)
    env = Monitor(env, f"./ppo/{state}")  # record stats such as returns
    return env

callback = SaveOnBestTrainingRewardCallbackCustom(check_freq=10000, log_dir=f"./ppo/{state}")

env = DummyVecEnv([make_env])
env = VecVideoRecorder(env, f"videos/{run.id}/{state}", record_video_trigger=lambda x: x % 10000 == 0, video_length=200)
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}/{state}")
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=[
        WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{run.id}/{state}",
            verbose=2,
        ),
        callback
    ]
)

model.save(f"models/{run.id}/{state}")

env.close()
run.finish()