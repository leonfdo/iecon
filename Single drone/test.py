import gym
from sb3_contrib import RecurrentPPO
import os
from testing_env import *


models_dir="C:\\Users\\leonf\\OneDrive\\Documents\\SUTD\\Single_Multi_drone\\Single drone\\models\\PPO_1"
#model_path=f"{models_dir}/13300000"

visualization_mode='human'
# visualization_mode=None


#best one 3800000

best_model=3800000

test=100
arr=[]
for j in range(0,15000000,100000):
    model_path=f"{models_dir}/{best_model}"
    win=0
    tot_steps=0
    for i in range(test):
        env=drone_env(render_mode=visualization_mode,num=i)
        obs=env.reset()
        #obs=env.reset()
        #obs=env.reset()
        #model=PPO.load(model_path,env=env)
        model = RecurrentPPO.load(model_path)
        lstm_states = None
        num_envs = 1

        #vec_env=model.get_env()
        episode_starts = np.ones((num_envs,), dtype=bool)

        dones=False
        while not dones:
            action, lstm_states = model.predict(obs, state=lstm_states, episode_start=episode_starts, deterministic=True)
            obs, rewards, dones, info,reach,steps = env.step(action)
            episode_starts = dones
        if reach:
            tot_steps+=steps
            win+=1
    per=(win/test)*100
    if win==0:
        win=1
    arr.append(per)
    print(f"the percentage win for {j} is : {per}   Average nu of steps : {tot_steps/win}" )
