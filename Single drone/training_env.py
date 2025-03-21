import pygame
import time
import gym
import random
import csv
import numpy as np
from gym import spaces
from stable_baselines3 import PPO
from PIL import Image

screen_width = 700
screen_height = 700
resolution = 10
cols = int(screen_width / resolution)
rows = int(screen_height / resolution)
FPS = 10



cells = np.zeros((rows, cols), dtype=int)

def map_grid(n):
    file_name="C:\\Users\\leonf\\OneDrive\\Documents\\SUTD\\Single_Multi_drone\\maps_multi\\train_map"+str(n)+".csv"
    with open(file_name,"r") as file:
        reader=csv.reader(file)
        for i,row in enumerate(reader):
            for j,value in enumerate(row):
                if value=="0":
                    cells[i][j]=0
                elif value=="1":
                    cells[i][j]=1
                elif value=="*":
                    cells[i][j]=0
                    target_pos=[i,j]
                elif value=="+":
                    cells[i][j]=0
                    drone_cord=[i,j]
    return target_pos,drone_cord,i+1,j+1



def get_distance(target_pos, signal_pos):
    return (((target_pos[0] - signal_pos[0]) ** 2 + (target_pos[1] - signal_pos[1]) ** 2) ** 0.5)*resolution


def get_signal_strength(target_pos, signal_pos,RADIUS_TARGET_WEAK_SIGNAL):
    distance = get_distance(target_pos, signal_pos)
    #print(distance)
    if distance > RADIUS_TARGET_WEAK_SIGNAL:
        return 0
    return 5 - (distance / RADIUS_TARGET_WEAK_SIGNAL) * 5

    
    

class drone_env(gym.Env):
    def __init__(self,render_mode,num):
        self.reset_num=0
        self.random_num=num
        self.num=0
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8)
        self.observation_space = gym.spaces.Dict({
            "surrounding_cells": gym.spaces.MultiDiscrete([2] * 8),
            "Searched_cells":gym.spaces.MultiDiscrete([5] * 8),
             "signal_strength": gym.spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32)
        })


    def drone_move(self,snake_dir,drone_pos,cells):
    #snake_dir=int(snake_dir* 5)

        if(self.prev==0):
            self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]=0.2
        elif(self.prev==0.2):
            self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]=0.3
        elif(self.prev==0.3):
            self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]=0.4
        
        if snake_dir == 0:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=0.2 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            #cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=2 if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==0 else cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0] -= 1
            
        elif snake_dir == 1:
            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0] += 1
            
        elif snake_dir == 2:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=0.2 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            #cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][1] += 1
            
        elif snake_dir == 3:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=0.2 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            #cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][1] -= 1
            
        elif snake_dir == 4:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=0.2 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]+=1
            drone_pos[0][1] -=1
            
        elif snake_dir == 5:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=0.2 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            #cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=0.2 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]-=1
            drone_pos[0][1]-=1
            
        elif snake_dir == 6:
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]=(0.2 if  self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]])
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]=(0.2 if self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1])
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=(0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=(0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=(0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]])
            #cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]=(0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)])
            self.obs_cells[(drone_pos[0][0] + 1)][(drone_pos[0][1] + 1)]=(0.2 if self.obs_cells[(drone_pos[0][0] + 1)][(drone_pos[0][1] + 1)]==0 else self.obs_cells[(drone_pos[0][0] + 1)][(drone_pos[0][1] + 1)])
            drone_pos[0][0]-=1
            drone_pos[0][1]+=1
            
        elif snake_dir == 7:
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]=(0.2 if  self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]])
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]=(0.2 if self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1])
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=(0.2 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=(0.2 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=(0.2 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]])
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)]=(0.2 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)])
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]=(0.2 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)])
            #cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]+=1
            drone_pos[0][1]+=1


        if (drone_pos[0][0] < 0 or drone_pos[0][0] > rows-1 ) or (drone_pos[0][1] < 0 or drone_pos[0][1] > cols-1):
            colision = True
        else:
            colision = (cells[drone_pos[0][0]][drone_pos[0][1]]==1)
        return colision
    

    def step(self, action):
        self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]=0
        colision = self.drone_move(action,self.drone_pos,cells)
        self.signal_strength = get_signal_strength(self.target_pos, self.drone_pos[0],self.RADIUS_TARGET_WEAK_SIGNAL)
        self.prev=self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]
        distance = get_distance(self.target_pos, self.drone_pos[0])
        reward_t=0
        reward_b=0
        if colision:
            reward_a=-50
            self.done=True
        else:
            #reward_a=0
            if self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]==0:
                reward_a=2
            elif (self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]==0.2):
                reward_a=0
            elif (self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]==0.3):
                reward_a=-1
            else:
                reward_a=-4
                

        #print(cells[self.drone_pos[0][0]][self.drone_pos[0][1]])
        if distance <= self.RADIUS_TARGET_HIGH_SIGNAL:
            self.done=True
            print(f"Target reached hoooraaaaaay")
            reward_b = 1000-reward_a


        self.reward = reward_a + reward_b    
        #self.prev_reward = self.reward
        self.info = {}
        self.steps_taken += 1

        if self.max_steps<self.steps_taken:
            self.done=True

        if distance <= self.RADIUS_TARGET_WEAK_SIGNAL:
            #print(self.target_pos)
            self.obs_cells[self.target_pos[0],self.target_pos[1]]=0.9


        if cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]==1: self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]= 1
        if cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]==1: self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]= 1
        if cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]==1: self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]= 1
        if cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]==1: self.obs_cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]=1
        if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==1: self.obs_cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=1
        if cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]==1: self.obs_cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]=1
        if cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]==1: self.obs_cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]=1
        if cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]==1: self.obs_cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]=1
        self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]=0.8


        if self.render_mode == 'human':
            self.render()


        self.observation = {
            "surrounding_cells": [
                #cells[self.drone_row + 1][self.drone_col - 1],
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]]==1,
                #cells[self.drone_row + 1][self.drone_col + 1],
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]-1]==1,
                cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]==1,
                cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]==1,
                cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==1,
                cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)%60]==1,
                cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)%60]==1,
                cells[(self.drone_pos[0][0] + 1)%60][(self.drone_pos[0][1] + 1)%60]==1
                ],
            "Searched_cells":[
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]],
                #cells[self.drone_row + 1][self.drone_col + 1],
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]-1],
                cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1],
                cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1],
                cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]],
                cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)%60],
                cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)%60],
                cells[(self.drone_pos[0][0] + 1)%60][(self.drone_pos[0][1] + 1)%60]
            ],
            "signal_strength": [self.signal_strength]
        }
        
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.done = False
        self.inside=False
        self.prev=0
        self.drone_dir = 0
        self.reset_num+=1
        #self.cells = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            for j in range(cols):
                if cells[i][j] == 2:
                   cells[i][j] = 0
                elif cells[i][j]==3:
                    cells[i][j]=0
        #self.random_num=1
        if self.reset_num%50==0:
            self.random_num+=1
            #print(f"is the random nu :{self.random_num} ****************************************************************************************")
            self.random_num=self.random_num%71
        #print(self.random_num,self.reset_num)
        print(f"the random nu :{self.random_num}********************************************************")
        self.target_pos,self.drone_cord,self.row,self.col=map_grid(self.random_num+1)
        self.obs_cells = np.zeros((self.row, self.col), dtype=float)
        
        self.RADIUS_TARGET_WEAK_SIGNAL= 60
        self.RADIUS_TARGET_MEDIUM_SIGNAL = 30
        self.RADIUS_TARGET_HIGH_SIGNAL = 10
        #print(self.RADIUS_TARGET_WEAK_SIGNAL)
        #print(self.target_pos,self.drone_cord)
        self.target_row = self.target_pos[0]
        self.target_col = self.target_pos[1]
        #self.target_pos = [self.target_row, self.target_col]
        self.drone_pos = []
        self.drone_pos.append(self.drone_cord)
        self.drone_row = self.drone_pos[0][0]
        self.drone_col = self.drone_pos[0][1]
        self.obs_cells[self.drone_row][self.drone_col]=0.8
        self.signal_strength = get_signal_strength(self.target_pos, self.drone_pos[0],self.RADIUS_TARGET_WEAK_SIGNAL)
        cells[self.drone_row][self.drone_col]=2
        #print(f"row {self.drone_row} and coloumn {self.drone_col}")
    

        #self.observation = np.array(self.observation)
        self.reward = 0
        self.prev_reward = 0
        
        self.max_steps = 10000
        #print(f"number of maximum steps {self.max_steps}")
        self.steps_taken = 0
        '''
        if (self.num%200==0):
            self.render_mode='human'
        else:
            self.render_mode=None
        '''
        
        self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]= (1 if  cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]==1 else 0)
        self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]= (1 if cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]==1 else 0 )
        self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]= (1 if cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]==1 else 0)
        self.obs_cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]=(1 if cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]==1 else 0)
        self.obs_cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=(1 if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==1 else 0)
        self.obs_cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]=(1 if cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]==1 else 0)
        self.obs_cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]=(1 if cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]==1 else 0)
        self.obs_cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]=(1 if cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]==1 else 0)

        
        #self.genai()

        self.observation = {
            "surrounding_cells": [
                #cells[self.drone_row + 1][self.drone_col - 1],
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]]==1,
                #cells[self.drone_row + 1][self.drone_col + 1],
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]-1]==1,
                cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]==1,
                cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]==1,
                cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==1,
                cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)%60]==1,
                cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)%60]==1,
                cells[(self.drone_pos[0][0] + 1)%60][(self.drone_pos[0][1] + 1)%60]==1
                ],
            "Searched_cells":[
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]],
                #cells[self.drone_row + 1][self.drone_col + 1],
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]-1],
                cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1],
                cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1],
                cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]],
                cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)%60],
                cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)%60],
                cells[(self.drone_pos[0][0] + 1)%60][(self.drone_pos[0][1] + 1)%60]
            ],  
            "signal_strength": [self.signal_strength]
        }


        

        if self.render_mode == 'human':
            pygame.init()
            self.display = pygame.display.set_mode((screen_width, screen_height))
            self.background_image = pygame.image.load("camouflage.png").convert()
            self.background_image = pygame.transform.scale(self.background_image, (screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.render()
        self.num+=1
     
    
        return self.observation

    def render(self, render_mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.display.blit(self.background_image, (0, 0))

        
        for i in range(self.row):
            for j in range(self.col):
                if self.obs_cells[i][j] == 1:
                    pygame.draw.rect(self.display, (0, 0, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j]==0.2:
                    pygame.draw.rect(self.display, (255, 255, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j]==0.3:
                    pygame.draw.rect(self.display, (0, 255, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j]==0.4:
                    pygame.draw.rect(self.display, (255, 0, 255), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j] == 0.8:
                    pygame.draw.rect(self.display, (255, 0, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j] == 0:
                    pygame.draw.rect(self.display, (255, 255, 255), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j] == 0.9:
                    pygame.draw.rect(self.display, (0,0, 255), (j * resolution, i * resolution, resolution, resolution))
                else:
                    pygame.draw.rect(self.display, (100, 0, 100), (j * resolution, i * resolution, resolution, resolution))
                    
                    
        #pygame.draw.rect(self.display, (0, 0, 255),
                         #(self.target_col * resolution, self.target_row * resolution, resolution, resolution))
        #pygame.draw.circle(self.display, (224, 255, 255),
                           #(self.target_col * resolution + resolution // 2,
                            #self.target_row * resolution + resolution // 2), self.RADIUS_TARGET_HIGH_SIGNAL, 2)
        #pygame.draw.circle(self.display, (224, 255, 255),
                           #(self.target_col * resolution + resolution // 2,
                           # self.target_row * resolution + resolution // 2), self.RADIUS_TARGET_MEDIUM_SIGNAL, 2)
        #pygame.draw.circle(self.display, (224, 255, 255),
                           #(self.target_col * resolution + resolution // 2,
                           # self.target_row * resolution + resolution // 2), self.RADIUS_TARGET_WEAK_SIGNAL, 2)

        #pygame.draw.rect(self.display, (255, 0, 0),
                         #(self.drone_pos[0][1] * resolution, self.drone_pos[0][0] * resolution, resolution, resolution))

                           
        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(FPS)
        if self.done:
            time.sleep(0.5)
    def close(self):
        pygame.quit()



env=drone_env(render_mode='human',num=6)
#model=PPO.load("C:\\Users\\leonf\\OneDrive\\Documents\\SUTD\\Myenv\\Training\\saved_models.zip",env=env)
observation=env.reset()
print(observation)
#print(f"the row and coloumn are : {env.drone_row} , {env.drone_col}")


'''
print(observation)
eps=10
for i in range(eps):
    done=False
    while not done:
        #action=model.predict(observation)
        #action=env.action_space.sample()
        observation,reward,done,info=env.step(2)
        #print(env.drone_pos)
        print(observation)
        #print(i,done)
    #print("epp is over %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
    observation=env.reset()
env.close()

'''
