import pygame
import time
import gym
import random
import csv
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

screen_width = 700
screen_height = 700
resolution = 10
cols = int(screen_width / resolution)
rows = int(screen_height / resolution)
FPS = 10




#prob_obstacle = 0.1

RADIUS_TARGET_WEAK_SIGNAL = 60
RADIUS_TARGET_MEDIUM_SIGNAL = 30
RADIUS_TARGET_HIGH_SIGNAL = 10
Drone_radius=40



cells = np.zeros((rows, cols), dtype=int)

def map_grid(n,drone_pos):
    file_name="E://Billy//maps_multi//train_map"+str(n)+".csv"
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
                    drone_pos.append([i,j])
    return target_pos




#map_grid(3)




def get_distance(target_pos, signal_pos):
    return (((target_pos[0] - signal_pos[0]) ** 2 + (target_pos[1] - signal_pos[1]) ** 2) ** 0.5)*resolution


def get_signal_strength(target_pos, signal_pos):
    distance = get_distance(target_pos, signal_pos)
    #print(distance)
    if distance > RADIUS_TARGET_WEAK_SIGNAL:
        return 0
    return 5 - (distance / RADIUS_TARGET_WEAK_SIGNAL) * 5


class drone_env(gym.Env):
    def __init__(self,render_mode,num):
        self.reset_num=0
        self.random_num=num
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(8)
        self.observation_space = gym.spaces.Dict({
            "surrounding_cells": gym.spaces.MultiDiscrete([2] * 8),
            "Searched_cells":gym.spaces.MultiDiscrete([5] * 8),
            "signal_strength": gym.spaces.Box(low=0, high=5, shape=(1,), dtype=np.float32),
            "neighbour_drone" : gym.spaces.MultiDiscrete([2]*4)
        })
        

    def reached(self,target_pos,signal_pos):
        row=(target_pos[0]+signal_pos[0])
        col=(target_pos[1]+signal_pos[1])
        #print(target_pos,signal_pos,row,col)
        if(row%2==0):
            if(col%2==1):
                t=int(row/2)
                c=col//2
                return (cells[t,c]==1 or cells[t,c+1]==1)
            else:
                t=int(row/2)
                c=int(col/2)
                return cells[t,c]==1
        else:
            if(col%2==0):
                t=row//2
                c=int(col/2)
                return (cells[t,c]==1 or cells[t+1,c]==1)
            

    def drone_move(self,snake_dir,i,cells):
    #snake_dir=int(snake_dir* 5)
        if snake_dir == 0:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            #cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=2 if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==0 else cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][0] -= 1
            
        elif snake_dir == 1:
            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][0] += 1
            
        elif snake_dir == 2:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            #cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][1] += 1
            
        elif snake_dir == 3:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            #cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][1] -= 1
    
        elif snake_dir == 4:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][0]+=1
            self.drone_pos[i][1] -=1
            
        elif snake_dir == 5:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            #cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][0]-=1
            self.drone_pos[i][1]-=1
            
        elif snake_dir == 6:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            #cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]=2 if cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==0 else cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
            self.drone_pos[i][0]-=1
            self.drone_pos[i][1]+=1
            
        elif snake_dir == 7:
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]=2 if  cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]
            cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]=2 if cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==0 else cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]
            cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]=2 if cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==0 else cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]
            cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]=2 if cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==0 else cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]
            cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]
            cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]=2 if cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==0 else cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]
            #cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            self.drone_pos[i][0]+=1
            self.drone_pos[i][1]+=1


        if (self.drone_pos[i][0] < 0 or self.drone_pos[i][0] > rows-1 ) or (self.drone_pos[i][1] < 0 or self.drone_pos[i][1] > cols-1):
            colision = True
        else:
            colision = (cells[self.drone_pos[i][0]][self.drone_pos[i][1]]==1) or (any(self.drone_pos[j] == self.drone_pos[i] for j in range(len(self.drone_pos)) if j != i))

        return colision

    def neighbour(self,drone_pos,j):
        arr=[0,0,0,0]
        for i in range(len(self.drone_pos)):
            if(i==j):
                continue
            if(get_distance(drone_pos,self.drone_pos[i])>Drone_radius):
                return arr
            else:
                #print(drone_pos,self.drone_pos[i])
                if(drone_pos[0]<=self.drone_pos[i][0]):
                    if(drone_pos[1]<=self.drone_pos[i][1]):
                        arr[0]=1
                    else:
                        arr[3]=1
                else:
                    if(drone_pos[1]<=self.drone_pos[i][1]):
                        arr[1]=1
                    else:
                        arr[2]=1
            return arr

                    
                
    def step(self, action):
        self.reward=0
        self.observation=[]

        for i in range(len(self.drone_pos)):
            colision = self.drone_move(action[i],i,cells)
            self.signal_strength = get_signal_strength(self.target_pos, self.drone_pos[i])
            distance = get_distance(self.target_pos, self.drone_pos[i])

            reward_b=0
            if colision:
                reward_a=-50
                self.done=True
            else:
                #reward_a=0
                if cells[self.drone_pos[i][0]][self.drone_pos[i][1]]==0:
                    reward_a=2
                    reward_b = self.signal_strength*75
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1]]=2
                elif (cells[self.drone_pos[i][0]][self.drone_pos[i][1]]==2):
                    reward_a=0
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1]]+=1
                elif (cells[self.drone_pos[i][0]][self.drone_pos[i][1]]==3):
                    reward_a=-1
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1]]+=1
                else:
                    reward_a=-4
                
                
        #print(colision)
        
            if distance <= RADIUS_TARGET_HIGH_SIGNAL:
                self.done=True
                print(f"Target reached hoooraaaaaay")
                reward_b = 1000-reward_a
            #print(reward_a,reward_b)    
            self.reward_drone = reward_a + reward_b
            self.reward+=self.reward_drone

            #cells[self.drone_pos[i][0]][self.drone_pos[i][1]]=2
            #self.observation.append(self.observation_drone)
            
        for i in range(len(self.drone_pos)):
            self.signal_strength = get_signal_strength(self.target_pos, self.drone_pos[i])
            if(i==0):
                self.observation_drone = {
                    "surrounding_cells": [
                    #cells[self.drone_row + 1][self.drone_col - 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==1,
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==1,
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==1,
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==1
                    ],
                    "Searched_cells":[
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]],
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1],
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]],
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60],
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60],
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
                ],
                    "signal_strength": [self.signal_strength],
                    "neighbour_drone": self.neighbour(self.drone_pos[i],i)
                }
            else:
                self.observation_drone = {
                    "surrounding_cells": [
                    #cells[self.drone_row + 1][self.drone_col - 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==1,
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==1,
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==1,
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==1
                    ],
                    "Searched_cells":[
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]],
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1],
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]],
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60],
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60],
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
                ],
                    "signal_strength": [self.signal_strength]     
                }
                

            self.observation.append(self.observation_drone)
            
        #self.reward = reward_a + reward_b    
        #self.prev_reward = self.reward
        self.info = {}
        self.steps_taken += 1

        if self.render_mode == 'human':
            self.render()
 
        #print(f"row {self.drone_pos[0][0]} and coloumn {self.drone_pos[0][1]}")
        #print(f"row : {self.drone_pos[0][0]} coloumn : {self.drone_pos[0][1]}")
        #self.drone_pos[0][0]=self.drone_pos[0][0]%15
        #self.drone_pos[0][1]=self.drone_pos[0][1]%29
        #print(f"row : {self.drone_pos[0][0]} coloumn : {self.drone_pos[0][1]}")
        
        
        
        #self.observation = np.array(self.observation)
        if self.max_steps<self.steps_taken:
            self.done=True
        #print(self.observation[0],self.reward,self.done)
        if(any(get_distance(self.drone_pos[i],self.drone_pos[j])<Drone_radius for j in range(len(self.drone_pos)) if j != i)):
            self.reward-=(2)
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.done = False
        self.drone_dir = 0
        self.reset_num+=1
        #self.cells = np.zeros((rows, cols), dtype=int)
        for i in range(rows):
            for j in range(cols):
                if cells[i][j] == 2:
                   cells[i][j] = 0
        #self.random_num=1
        if self.reset_num%50==0:
            self.random_num+=1
            self.random_num=self.random_num%71
        #print(self.random_num)
        #print(self.random_num,self.reset_num)
        print(f"the random nu :{self.random_num+1}********************************************************")
        self.drone_pos = []
        self.target_pos=map_grid(self.random_num+1,self.drone_pos)
        #print(self.drone_pos)
        #print(self.target_pos,self.drone_cord)
        self.target_row = self.target_pos[0]
        self.target_col = self.target_pos[1]
        #self.target_pos = [self.target_row, self.target_col]
        #self.drone_pos = []
        #self.drone_pos.append(self.drone_cord)
        self.observation=[]
        
        for i in range(len(self.drone_pos)):
            self.drone_row = self.drone_pos[i][0]
            self.drone_col = self.drone_pos[i][1]
            self.signal_strength = get_signal_strength(self.target_pos, self.drone_pos[i])
            cells[self.drone_row][self.drone_col]=2
            #print(f"row {self.drone_row} and coloumn {self.drone_col}")
            #print(self.drone_pos[i])
            #print([self.drone_pos[i][0] - 1,self.drone_pos[i][1]+1] in self.drone_pos)
            if(i==0):
                self.observation_drone = {
                    "surrounding_cells": [
                    #cells[self.drone_row + 1][self.drone_col - 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==1,
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==1,
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==1,
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==1
                    ],
                    "Searched_cells":[
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]],
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1],
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]],
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60],
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60],
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
                ],
                    "signal_strength": [self.signal_strength],
                    "neighbour_drone": self.neighbour(self.drone_pos[i],i)       
                }
            else:
                self.observation_drone = {
                    "surrounding_cells": [
                    #cells[self.drone_row + 1][self.drone_col - 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]]==1,
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1]==1,
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1]==1,
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]]==1,
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60]==1,
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]==1
                    ],
                    "Searched_cells":[
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]],
                    #cells[self.drone_row + 1][self.drone_col + 1],
                    cells[(self.drone_pos[i][0] + 1)%60][self.drone_pos[i][1]-1],
                    cells[self.drone_pos[i][0]][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0]-1][self.drone_pos[i][1] - 1],
                    cells[self.drone_pos[i][0] - 1][self.drone_pos[i][1]],
                    cells[self.drone_pos[i][0]-1][(self.drone_pos[i][1] + 1)%60],
                    cells[self.drone_pos[i][0]][(self.drone_pos[i][1] + 1)%60],
                    cells[(self.drone_pos[i][0] + 1)%60][(self.drone_pos[i][1] + 1)%60]
                ],
                    "signal_strength": [self.signal_strength]       
                }
            self.observation.append(self.observation_drone)

        #self.observation = np.array(self.observation)
        self.reward = 0
        self.prev_reward = 0
        
        self.max_steps = 10000
        self.steps_taken = 0

        if self.render_mode == 'human':
            pygame.init()
            self.display = pygame.display.set_mode((screen_width, screen_height))
            self.background_image = pygame.image.load("camouflage.png").convert()
            self.background_image = pygame.transform.scale(self.background_image, (screen_width, screen_height))
            self.clock = pygame.time.Clock()
            self.render()

        return self.observation

    def render(self, render_mode='human'):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self.display.blit(self.background_image, (0, 0))
        for i in range(rows):
            for j in range(cols):
                if cells[i][j] == 1:
                    pygame.draw.rect(self.display, (0, 0, 0), (j * resolution, i * resolution, resolution, resolution))
                elif cells[i][j]==2:
                    pygame.draw.rect(self.display, (255, 255, 0), (j * resolution, i * resolution, resolution, resolution))
                elif cells[i][j]==3:
                    pygame.draw.rect(self.display, (0, 255, 0), (j * resolution, i * resolution, resolution, resolution))
                elif cells[i][j]==4:
                    pygame.draw.rect(self.display, (255, 0, 255), (j * resolution, i * resolution, resolution, resolution))
                    

        pygame.draw.rect(self.display, (20, 120, 255),
                         (self.target_col * resolution, self.target_row * resolution, resolution, resolution))
        pygame.draw.circle(self.display, (224, 255, 255),
                           (self.target_col * resolution + resolution // 2,
                            self.target_row * resolution + resolution // 2), RADIUS_TARGET_HIGH_SIGNAL, 2)
        pygame.draw.circle(self.display, (224, 255, 255),
                           (self.target_col * resolution + resolution // 2,
                            self.target_row * resolution + resolution // 2), RADIUS_TARGET_MEDIUM_SIGNAL, 2)
        pygame.draw.circle(self.display, (224, 255, 255),
                           (self.target_col * resolution + resolution // 2,
                            self.target_row * resolution + resolution // 2), RADIUS_TARGET_WEAK_SIGNAL, 2)
        for i in range(len(self.drone_pos)):
            if(i==0):
                #print("came 1")
                pygame.draw.rect(self.display, (0, 0, 255),
                                 (self.drone_pos[i][1] * resolution, self.drone_pos[i][0] * resolution, resolution, resolution))
            else:
                #print("came 2")
                pygame.draw.rect(self.display, (255,0, 0),
                                 (self.drone_pos[i][1] * resolution, self.drone_pos[i][0] * resolution, resolution, resolution))
                
            pygame.draw.circle(self.display, (224, 255, 255),
                           (self.drone_pos[i][1] * resolution + resolution // 2,
                            self.drone_pos[i][0] * resolution + resolution // 2),Drone_radius, 2)
        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(FPS)
        if self.done:
            time.sleep(0.5)
    def close(self):
        pygame.quit()

       
'''
env=drone_env(render_mode='human',num=1)
#model=PPO.load("C:\\Users\\leonf\\OneDrive\\Documents\\SUTD\\Myenv\\Training\\saved_models.zip",env=env)
observation=env.reset()
print(observation)
#print(f"the row and coloumn are : {env.drone_row} , {env.drone_col}")

'''
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
