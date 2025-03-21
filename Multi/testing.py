from multi_drone_testing_env import *
from sb3_contrib import RecurrentPPO

visualization_mathod="human"
#visualization_method=None


for j in range(100000,10000000,50000):
    models_dir1="multi_models/PPO_LSTM_test5"
    model_path1=f"{models_dir1}/{j}"

    models_dir2="multi_models/PPO_LSTM_test5"
    model_path2=f"{models_dir2}/{j}"
    tot=0
    tot_steps=0
    for i in range(100):
        env=drone_env(render_mode=visualization_mathod,num=i)
        #print(i)
        obs=env.reset()
        #print(obs)
        model1 = RecurrentPPO.load(model_path2)
        model2 = RecurrentPPO.load(model_path2)

        lstm_states1 = None
        lstm_states2=None

        num_envs = 1

        episode_starts1 = np.ones((num_envs,), dtype=bool)
        episode_starts2 = np.ones((num_envs,), dtype=bool)

        dones=False

        while not dones:
            action1, lstm_states1 = model1.predict(obs[1], state=lstm_states1, episode_start=episode_starts1, deterministic=True)
            action2, lstm_states2 = model2.predict(obs[0], state=lstm_states2, episode_start=episode_starts2, deterministic=True)
            #print(f"{action1} , {action2}")
            #print("*********************************************************************")
            obs, rewards, dones, info,finish,steps = env.step([action2,action1])
            episode_starts1=dones
            episode_starts2=dones
        if finish:
            tot+=1
            tot_steps+=steps
    print(f"for {j} the are success rate : {tot} the number of steps:{tot_steps/tot if tot!=0 else 1}")
    
