import pygame
import time
import gym
import random
import csv
import numpy as np
from gym import spaces
from stable_baselines3 import PPO

from groq import Groq
import base64
import cv2 
import pandas as pd
import matplotlib.pyplot as plt
import os
from PIL import Image
from io import BytesIO 
import json

#set environment variables
os.environ['GROQ_API_KEY'] = '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'

screen_width = 700
screen_height = 700
resolution = 10
cols = int(screen_width / resolution)
rows = int(screen_height / resolution)
FPS = 10
colour_map = {0: [200, 200, 200], 100: [255, 255, 0], 125: [255, 165, 0], 150: [255, 0, 0], 50: [0, 0, 255], 200: [255, 0, 255], 255: [0, 0, 0]}




cells = np.zeros((rows, cols), dtype=int)

def map_grid(n):
    file_name="maps_multi\\train_map"+str(n)+".csv"
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

# Function to encode the image
def encode_image(img):
#   with open(image_path, "rb") as image_file:
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)  # Convert NumPy array to PIL Image
    
    # Save the image in memory
    buffered = BytesIO()
    img.save(buffered, format="PNG")  # Convert to PNG format
    return base64.b64encode(buffered.getvalue()).decode("utf-8")
    # return base64.b64encode(img).decode('utf-8')

def colour_mapping(img):
    # do the colour mapping 
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j, 0] == 0:
                img[i, j] = colour_map[0]
            elif img[i, j, 0] == 50:
                img[i, j] = colour_map[50]
            elif img[i, j, 0] == 100:
                img[i, j] = colour_map[100]
            elif img[i, j, 0] == 125:
                img[i, j] = colour_map[125]
            elif img[i, j, 0] == 150:
                img[i, j] = colour_map[150]
            elif img[i, j, 0] == 200:
                img[i, j] = colour_map[200]
            elif img[i, j, 0] == 255:
                img[i, j] = colour_map[255]
    return img





def get_distance(target_pos, signal_pos):
    return (((target_pos[0] - signal_pos[0]) ** 2 + (target_pos[1] - signal_pos[1]) ** 2) ** 0.5)*resolution

    


def get_signal_strength(target_pos, signal_pos,RADIUS_TARGET_WEAK_SIGNAL):
    distance = get_distance(target_pos, signal_pos)
    #print(distance)
    if distance > RADIUS_TARGET_WEAK_SIGNAL:
        return 0
    return 5 - (distance / RADIUS_TARGET_WEAK_SIGNAL) * 5

client = Groq()

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
             "vision_pred": gym.spaces.MultiDiscrete([8])
        })

    def csv_to_img(self):
        # df = pd.D
        # img = df.to_numpy()
        img = self.obs_cells.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = colour_mapping(img)
        # breakpoint()
        # plt.imshow(img)
        # plt.show()
        return img[:,:, ::-1]


    def genai(self, img):
        #model eke methenin danne csv eke self.obs_cells

        # Path to your image
    # image_path = f"./processed_ulindu/model_2_step_10.png"

    # Getting the base64 string
        base64_image = encode_image(img)



        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": r"""The given image is a maze that a drone tries to explore. 
        In black - Obstacle(Wall). 
        In gray - Unexplored area. 
        In Yellow - Explored area (Drone has explored this area once).
        In Orange - Explored area (Drone has explored this area twice).
        In Red - Explored area (Drone has explored this area more than twice).
        In Blue - Drone's current position. 
        The drone can't go through walls.
        Guide the drone to where it needs to go to explore the rest of the maze. State the most globally unexplored direction relative to the drone's current position.
        Do not mistakenly say directions relative to the center of the image. It should always be relative to the drone's current position.
        The possible directions are North, South, East, West, North-East, North-West, South-East, South-West. 
        Provide a concise reasoning (max 20 words). 
        Only the direction is needed. 
        Use json format.

        ex: {"reasoning":"There are grey ares in East and North-West areas relative to the drone. But the mojority of the grey areas are in East", "direction": "East"}
        """},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model="llama-3.2-90b-vision-preview",
            temperature = 0,
            top_p=1,
            response_format={"type": "json_object"},
            
        )

        pred = json.loads(chat_completion.choices[0].message.content)
        #print(pred)
        # breakpoint()

        if pred["direction"].lower() == "north":
            dir = 0
        elif pred["direction"].lower() == "north-east":
            dir = 1
        elif pred["direction"].lower() == "east":
            dir = 2
        elif pred["direction"].lower() == "south-east":
            dir = 3
        elif pred["direction"].lower() == "south":
            dir = 4
        elif pred["direction"].lower() == "south-west":
            dir = 5
        elif pred["direction"].lower() == "west":
            dir = 6
        elif pred["direction"].lower() == "north-west":
            dir = 7
        else:
            print("Invalid direction")
            dir = -1
        return dir
        

        
    def drone_move(self,snake_dir,drone_pos,cells):
    #snake_dir=int(snake_dir* 5)

        if(self.prev==0):
            self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]=100
        elif(self.prev==100):
            self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]=125
        elif(self.prev==125):
            self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]=150
    
        if snake_dir == 0:

            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=100 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            #cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=2 if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==0 else cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            #cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=2 if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==0 else cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0] -= 1

            
        elif snake_dir == 1:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]

            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0] += 1

            
            
        elif snake_dir == 2:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=100 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            #cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            #cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][1] += 1

            
        elif snake_dir == 3:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=100 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            #cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]

            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            #cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][1] -= 1

            
            
        elif snake_dir == 4:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=100 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]

            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            #cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]+=1
            drone_pos[0][1] -=1

            
            
        elif snake_dir == 5:
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=100 if  self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            #cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=100 if self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else self.obs_cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]


            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            #cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]-=1
            drone_pos[0][1]-=1

            
            
        elif snake_dir == 6:
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]=(100 if  self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]])
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]=(100 if self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1])
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=(100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=(100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=(100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]])
            #cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]=(100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)])
            self.obs_cells[(drone_pos[0][0] + 1)][(drone_pos[0][1] + 1)]=(100 if self.obs_cells[(drone_pos[0][0] + 1)][(drone_pos[0][1] + 1)]==0 else self.obs_cells[(drone_pos[0][0] + 1)][(drone_pos[0][1] + 1)])

            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            #cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]-=1
            drone_pos[0][1]+=1

            
            
        elif snake_dir == 7:
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]=(100 if  self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]])
            self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]=(100 if self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1]==0 else self.obs_cells[(drone_pos[0][0] + 1)][drone_pos[0][1]-1])
            self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]=(100 if self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=(100 if self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else self.obs_cells[drone_pos[0][0]-1][drone_pos[0][1] - 1])
            self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]=(100 if self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else self.obs_cells[drone_pos[0][0] - 1][drone_pos[0][1]])
            self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)]=(100 if self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)]==0 else self.obs_cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)])
            self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]=(100 if self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)]==0 else self.obs_cells[drone_pos[0][0]][(drone_pos[0][1] + 1)])
            #cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]

            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]=2 if  cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]]
            cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]=2 if cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]==0 else cells[(drone_pos[0][0] + 1)%60][drone_pos[0][1]-1]
            cells[drone_pos[0][0]][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]][drone_pos[0][1] - 1]
            cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]=2 if cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]==0 else cells[drone_pos[0][0]-1][drone_pos[0][1] - 1]
            cells[drone_pos[0][0] - 1][drone_pos[0][1]]=2 if cells[drone_pos[0][0] - 1][drone_pos[0][1]]==0 else cells[drone_pos[0][0] - 1][drone_pos[0][1]]
            cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]-1][(drone_pos[0][1] + 1)%60]
            cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]=2 if cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]==0 else cells[drone_pos[0][0]][(drone_pos[0][1] + 1)%60]
            #cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]=2 if cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]==0 else cells[(drone_pos[0][0] + 1)%60][(drone_pos[0][1] + 1)%60]
            drone_pos[0][0]+=1
            drone_pos[0][1]+=1
            

            

        if (drone_pos[0][0] < 0 or drone_pos[0][0] > rows-1 ) or (drone_pos[0][1] < 0 or drone_pos[0][1] > cols-1):
            colision = True
        else:
            colision = (cells[drone_pos[0][0]][drone_pos[0][1]]==1)
        return colision

    def step(self, action):
        self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]= self.prev
        colision = self.drone_move(action,self.drone_pos,cells)
        self.prev=self.obs_cells[(self.drone_pos[0][0])][self.drone_pos[0][1]]
       
        distance = get_distance(self.target_pos, self.drone_pos[0])

        if distance <= self.RADIUS_TARGET_WEAK_SIGNAL:
            self.obs_cells[self.target_pos[0],self.target_pos[1]]= 200

            
        reward_t=0
        reward_b=0
        reward_genai = 0
        
        if colision:
            reward_a=-50
            self.done=True
        else:
            #reward_a=0
            if cells[self.drone_pos[0][0]][self.drone_pos[0][1]]==0:
                reward_a=2
                cells[self.drone_pos[0][0]][self.drone_pos[0][1]]=2
            elif (cells[self.drone_pos[0][0]][self.drone_pos[0][1]]==2):
                reward_a=0
                cells[self.drone_pos[0][0]][self.drone_pos[0][1]]+=1
            elif (cells[self.drone_pos[0][0]][self.drone_pos[0][1]]==3):
                reward_a=-1
                cells[self.drone_pos[0][0]][self.drone_pos[0][1]]+=1
            else:
                reward_a=-4



        if (action) == self.vision_pred:
            reward_genai = 5
                

        #print(cells[self.drone_pos[0][0]][self.drone_pos[0][1]])
        if distance <= self.RADIUS_TARGET_HIGH_SIGNAL:
            self.done=True
            print(f"Target reached hoooraaaaaay")
            reward_b = 1000-reward_a



        self.reward = reward_a + reward_b + reward_genai 

        self.info = {}
        self.steps_taken += 1

        if cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]==1: self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]= 255
        if cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]==1: self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]= 255
        if cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]==1: self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]= 255
        if cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]==1: self.obs_cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]=255
        if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==1: self.obs_cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=255
        if cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]==1: self.obs_cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]=255
        if cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]==1: self.obs_cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]=255
        if cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]==1: self.obs_cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]=255
        self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1]]=50

        self.observation = {
            "surrounding_cells": [
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]]==1,
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
                cells[(self.drone_pos[0][0] + 1)%60][self.drone_pos[0][1]-1],
                cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1],
                cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1],
                cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]],
                cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)%60],
                cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)%60],
                cells[(self.drone_pos[0][0] + 1)%60][(self.drone_pos[0][1] + 1)%60]
            ],
            "vision_pred": self.vision_pred
        }
        

        if self.max_steps<self.steps_taken:
            self.done=True


        if self.render_mode == 'human':
            self.render()

        if(self.steps_taken%10==0):
            img = self.csv_to_img()
            self.vision_pred=self.genai(img)

        
        
        return self.observation, self.reward, self.done, self.info

    def reset(self):
        self.prev=0
        self.done = False
        self.inside=False
        self.drone_dir = 0
        self.reset_num+=1

        for i in range(rows):
            for j in range(cols):
                if cells[i][j] == 2:
                   cells[i][j] = 0
                elif cells[i][j]==3:
                    cells[i][j]=0

        if self.reset_num%50==0:
            self.random_num+=1
            #print(f"is the random nu :{self.random_num} ****************************************************************************************")
            self.random_num=self.random_num%61
        #print(self.random_num,self.reset_num)
        print(f"the random nu :{self.random_num}********************************************************")
        self.target_pos,self.drone_cord,self.row,self.col=map_grid(self.random_num+1)
        
        self.RADIUS_TARGET_WEAK_SIGNAL= 60
        self.RADIUS_TARGET_MEDIUM_SIGNAL = 30
        self.RADIUS_TARGET_HIGH_SIGNAL = 10
        self.target_row = self.target_pos[0]
        self.target_col = self.target_pos[1]
        self.drone_pos = []
        self.drone_pos.append(self.drone_cord)
        self.drone_row = self.drone_pos[0][0]
        self.drone_col = self.drone_pos[0][1]
        cells[self.drone_row][self.drone_col]=2

        #image
        #print(self.row,self.col)
        self.obs_cells = np.zeros((self.row, self.col), dtype=np.uint8)
        self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]= (255 if  cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]]==1 else 0)
        self.obs_cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]= (255 if cells[(self.drone_pos[0][0] + 1)][self.drone_pos[0][1]-1]==1 else 0 )
        self.obs_cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]= (255 if cells[self.drone_pos[0][0]][self.drone_pos[0][1] - 1]==1 else 0)
        self.obs_cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]=(255 if cells[self.drone_pos[0][0]-1][self.drone_pos[0][1] - 1]==1 else 0)
        self.obs_cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]=(255 if cells[self.drone_pos[0][0] - 1][self.drone_pos[0][1]]==1 else 0)
        self.obs_cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]=(255 if cells[self.drone_pos[0][0]-1][(self.drone_pos[0][1] + 1)]==1 else 0)
        self.obs_cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]=(255 if cells[self.drone_pos[0][0]][(self.drone_pos[0][1] + 1)]==1 else 0)
        self.obs_cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]=(255 if cells[(self.drone_pos[0][0] + 1)][(self.drone_pos[0][1] + 1)]==1 else 0)
        self.obs_cells[self.drone_row][self.drone_col]=50


        img = self.csv_to_img()
        self.vision_pred = self.genai(img)
        
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
            "vision_pred": self.vision_pred
        }

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
                if self.obs_cells[i][j] == 255 :
                    pygame.draw.rect(self.display, (0, 0, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j]==100:
                    pygame.draw.rect(self.display, (255, 255, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j]==125:
                    pygame.draw.rect(self.display, (0, 255, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j]==150:
                    pygame.draw.rect(self.display, (255, 0, 255), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j] == 50:
                    pygame.draw.rect(self.display, (255, 0, 0), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j] == 0:
                    pygame.draw.rect(self.display, (255, 255, 255), (j * resolution, i * resolution, resolution, resolution))
                elif self.obs_cells[i][j] == 200:
                    pygame.draw.rect(self.display, (0,0, 255), (j * resolution, i * resolution, resolution, resolution))
                else:
                    pygame.draw.rect(self.display, (100, 0, 100), (j * resolution, i * resolution, resolution, resolution))

                           
        pygame.display.flip()
        pygame.display.update()
        self.clock.tick(FPS)
        if self.done:
            time.sleep(0.5)
    def close(self):
        pygame.quit()







