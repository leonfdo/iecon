import gym
from stable_baselines3 import PPO
import os
from training_env import *
#from stable_baselines3.common.envs.observation_dict_wrapper import ObservationDictWrapper

models_dir="models/PPO_2"
logdir="logs"


if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)



env=drone_env(render_mode=None,num=0)
#env = ObservationDictWrapper(env)

#env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)


#env = VecFrameStack(env, n_stack=4)

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=logdir, use_sde=False)
#model=PPO('MultiInputPolicy',env,verbose=1,tensorboard_log=logdir)


env.reset()

timesteps=50000

for i in range(1,100):
    model.learn(total_timesteps=timesteps,reset_num_timesteps=False,tb_log_name="PPO_leo_test20")
    model.save(f"{models_dir}/{timesteps*i}")

env.close()

