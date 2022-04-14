import os

import retro
import gym
from PIL import Image
import cv2

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
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    
    # convert obs from numpy to opencv
    obs_reshaped = obs[0, 15:70, :, 0]
    img = np.array(obs_reshaped)
    #img = img.astype(np.uint8)
    gray = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)


    kernel_size = 5
    blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(blur_gray, low_threshold, high_threshold)
    cv2.imwrite(f"lane_detection/edges/{i}.png", edges)


    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 50  # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 50  # minimum number of pixels making up a line
    max_line_gap = 20  # maximum gap in pixels between connectable line segments
    line_image = np.copy(img) * 0  # creating a blank to draw lines on

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                        min_line_length, max_line_gap)
    if lines is not None:
        for line in lines:
            for x1,y1,x2,y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),2)

        lines_edges = cv2.addWeighted(img, 0.8, line_image, 1, 0)
        cv2.imwrite(f'lane_detection/hough/{i}.png', lines_edges)

    Z = gray.reshape((-1,3))
    # convert to np.float32
    Z = np.float32(Z)
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imwrite(f'lane_detection/segmentation/{i}.png', res2)

    env.render()
    if done:
      obs = env.reset()