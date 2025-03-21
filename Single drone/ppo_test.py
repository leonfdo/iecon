import gym
from stable_baselines3 import PPO
import os
from testing_env import *
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.vec_env import VecFrameStack

models_dir = "models\\PPO_2"
output_file = "results_3.txt"

with open(output_file, "w") as file:
    for j in range(50000, 6000000, 50000):
        finish = 0
        tot_steps = 0
        model_path = f"{models_dir}/{j}"
        for i in range(100):
            env = drone_env(render_mode=None, num=i)

            obs = env.reset()
            model = PPO.load(model_path, env=env)

            dones = False
            while not dones:
                action, _ = model.predict(obs)
                obs, rewards, dones, info = env.step(action)
                target_reach = env.finished
                steps = env.steps_taken

            if target_reach:
                finish += 1
                tot_steps += steps

        success_rate = (finish / 100) * 100
        average_steps = tot_steps / finish if finish > 0 else float('inf')
        
        # Print to console
        print(f"average number of steps : {average_steps}")
        print(f"for the {j} model the success rate is: {success_rate}")

        # Write to file
        file.write(f"for the {j} model the success rate is: {success_rate}% \n")

# This code will print the results to the console and also write them to a file named "results.txt".

