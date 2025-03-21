import gym
import os
from training_env import *
from sb3_contrib import RecurrentPPO




models_dir="models\\model1"

visualization_mode='human'
# visualization_mode=None

if not os.path.exists(models_dir):
    os.makedirs(models_dir)


env=drone_env(render_mode=visualization_mode,num=0)

#learning_rate=0.0003
#n_steps=128
#batch_size=128
#n_epochs=10
#gamma=0.99
#gae_lambda=0.95
#clip_range=0.2
#ent_coef=0.01
#vf_coef=0.5
#max_grad_norm=0.5
#seed=None

model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1)

env.reset()

timesteps=50000

for i in range(50000):
    model.learn(total_timesteps=timesteps,reset_num_timesteps=False)
    #vec_env = model.get_env()
    #mean_reward, std_reward = evaluate_policy(model, vec_env, n_eval_episodes=20, warn=False)
    #print(mean_reward)
    model.save(f"{models_dir}/{timesteps*i}")

env.close()



