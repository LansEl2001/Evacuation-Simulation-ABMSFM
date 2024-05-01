#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# Importing locals
from pygame.locals import *
from additional_functions import *
from environment import *
#get_ipython().run_line_magic('run', 'additional_functions.ipynb')
#get_ipython().run_line_magic('run', 'environment.ipynb')

# Creating evacuation object class
class Agent(object):
    def __init__(self,x,y):

        self.mass = 80  # random.uniform(40,90)
        self.radius = 20
        # random initialize a agent

        self.x = x
        self.y = y
        self.pos = np.array([self.x, self.y])
        # self.pos = np.array([10.0, 10.0])

        # Initialize dx and dy for herding
        self.dx = 0
        self.dy = 0

        self.aVelocityX = random.uniform(0, 16)
        self.aVelocityY = random.uniform(0, 16)
        self.aVelocity = np.array([self.aVelocityX, self.aVelocityY])
        # self.actualV = np.array([0.0, 0.0])

        self.dest = np.array([random.uniform(100, 1300), random.uniform(100 + 600, 1300 + 600)])
        self.direction = normalize(self.dest - self.pos)
        # self.direction = np.array([0.0, 0.0])

        self.dSpeed = 12
        self.dVelocity = self.dSpeed * self.direction

        self.acclTime = 0.5  # random.uniform(8,16) #10.0
        self.drivenAcc = (self.dVelocity - self.aVelocity) / self.acclTime

        self.bodyFactor = 120000
        self.F = 2000
        self.delta = 0.08 * 50  # random.uniform(0.8,1.6) #0.8 #0.08

        self.Goal = 0
        self.time = 0.0
        self.countcollision = 0

        #print('X and Y Position:', self.pos)
        #print('self.direction:', self.direction)
        
    #Exit Choice
    #rho max, kd is set in the main.py 
    def calculate_local_density(self, agents, exit_pos, exit_width):
        """
        agents: List of all agents to consider for density calculation
        exit_pos: Position of the exit (x, y)
        exit_width: Width of the exit
        
        return: Local density at the exit
        """
        lambda_factor = 1 #Regulatory factor
        radius = lambda_factor * exit_width
        num_agents = sum(1 for agent in agents if np.linalg.norm(agent.pos - np.array(exit_pos)) <= radius)
        area = np.pi * (radius ** 2)
        rho_exit = 2 * num_agents / area
        print(f"Radius: {radius}, Agents within radius: {num_agents}, exit pos: {exit_pos}")

        return rho_exit
    
    def calculate_exit_probabilities(self, agents, exits_info, pmax, k_d):
        """
        Calculate the normalized probability of choosing each exit.
        """
        densities = [self.calculate_local_density(agents, exit_info[0], exit_info[1]) for exit_info in exits_info]
        probabilities = [max(pmax, 10**6*density) ** k_d for density in densities]
        total = sum(probabilities)
        #print(f"Dens: {densities}")
        normalized_probabilities = [p / total if total > 0 else 0 for p in probabilities]
        #print(f"NProb: {normalized_probabilities}")
        
        return normalized_probabilities

    def velocity_force(self):  # function to adapt velocity
        deltaV = self.dVelocity - self.aVelocity
        if np.allclose(deltaV, np.zeros(2)):
            deltaV = np.zeros(2)
        return deltaV * self.mass / self.acclTime

    def f_ij(self, other):  # interaction with people
        r_ij = self.radius + other.radius
        d_ij = np.linalg.norm(self.pos - other.pos)
        e_ij = (self.pos - other.pos) / d_ij
        value = self.F * np.exp((r_ij - d_ij) / (self.delta)) * e_ij
        + self.bodyFactor * g(r_ij - d_ij) * e_ij

        if d_ij <= r_ij:
            self.countcollision += 1

        return value

    def f_ik_wall(self, wall):  # interaction with the wall in the room
        r_i = self.radius
        d_iw, e_iw = distance_agent_to_wall(self.pos, wall)
        value = -self.F * np.exp((r_i - d_iw) / self.delta) * e_iw  # Assume wall and people give same force
        + self.bodyFactor * g(r_i - d_iw) * e_iw
        return value

    def update_position(self):
        # for herding
        self.pos += self.direction * np.linalg.norm(self.aVelocity)
        self.x, self.y = self.pos

    def update_dest(self, agents, exits_info, pmax, k_d):
        exit_probabilities = self.calculate_exit_probabilities(agents, exits_info, pmax, k_d)
        chosen_exit_index_L1 = np.argmax(exit_probabilities)
        chosen_exit_index_L2 = np.argmax(exit_probabilities)-1

        if 1032.1/2 + buffer < self.pos[0] < 1917.9/2 + buffer - 885.8/4 and buffer < self.pos[1] < 751.1/2 + buffer: #123left
            self.dest = np.array([1032.1/2 + 200/2 + buffer + 50/2, 751.1/2 + buffer])
        elif 1917.9/2 + buffer - 885.8/4 < self.pos[0] < 1917.9/2 + buffer and buffer < self.pos[1] < 751.1/2 + buffer: #123right
            self.dest = np.array([1032.1/2 + 200/2 + 100/2 + 300/2 + 100/2 + buffer - 50/2,751.1/2 + buffer])

        elif 1032.1/2 + 885.8/2 + buffer < self.pos[0] < 1917.9/2 + 885.8/2 + buffer - 885.8/4 and buffer < self.pos[1] < 751.1/2 + buffer: #124left
            self.dest = np.array([1032.1 / 2 + 885.8 / 2 + 185.8 / 2 + buffer + 50/2, 751.1/2 + buffer])
        elif 1917.9/2 + 885.8/2 + buffer - 885.8/4 < self.pos[0] < 1917.9/2 + 885.8/2 + buffer and buffer < self.pos[1] < 751.1/2 + buffer: #124right
            self.dest = np.array([1032.1 / 2 + 885.8 / 2 + 185.8 / 2 + 100 / 2 + 300 / 2 + 50/2 + buffer,751.1/2 + buffer])

        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + 885.8 / 2 + (885.8 / 2) - 297/2 + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1 / 2 + 297 / 2 + buffer:  # 123-124 corridor
            self.dest = np.array([1917.9 / 2 + 885.8 / 2 + buffer, 751.1 / 2 + (297 / 2) / 2 + buffer])
        elif 1032.1 / 2 + 885.8 / 2 + (885.8 / 2) - 297/2 + buffer < self.pos[0] < 1032.1 / 2 + 885.8 / 2 + (885.8 / 2) + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1 / 2 + 297 / 2 + buffer:  # 124 exit box
            self.dest = np.array([1032.1 / 2 + 885.8 / 2 + (885.8 / 2) + buffer - 297/4, 751.1 / 2 + (297 / 2) + buffer + 500])


        #L1
        elif 735/2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1 / 2 + 297/2 + buffer:
            self.dest = np.array(exits_info[chosen_exit_index_L1][0])


        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1 / 2 + buffer + ( 885.8 / 2) / 2:  # room 118 top
            self.dest = np.array([735 / 2 + buffer, 751.1 / 2 + buffer + 150 / 2])
        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer + (885.8 / 2) / 2 < self.pos[1] < 751.1 / 2 + buffer + (885.8 / 2):  # room 118 bottom
            self.dest = np.array( [735 / 2 + buffer, 751.1 / 2 + buffer + (885.8 / 2) - 150 / 2])

        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer + 885.8 / 2 < self.pos[1] < 751.1 / 2 + buffer + (885.8 / 2) / 2 + 885.8 / 2:  # room 117 top
            self.dest = np.array([735 / 2 + buffer, 751.1 / 2 + buffer + 150 / 2 + 885.8 / 2])
        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer + (885.8 / 2) / 2 + 885.8 / 2 < self.pos[1] < 751.1 / 2 + buffer + (885.8 / 2) + 885.8 / 2:  # room 117 bottom
            self.dest = np.array( [735 / 2 + buffer, 751.1 / 2 + buffer + (885.8 / 2) - 150 / 2 + 885.8 / 2])

        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer + 2 * 885.8 / 2 < self.pos[1] < 751.1 / 2 + buffer + (885.8 / 2) / 2 + 2 * 885.8 / 2:  # room 116 top
            self.dest = np.array( [735 / 2 + buffer, 751.1 / 2 + buffer + 150 / 2 + 2 * 885.8 / 2])
        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer + (885.8 / 2) / 2 + 2 * 885.8 / 2 < self.pos[1] < 751.1 / 2 + buffer + (885.8 / 2) + 2 * 885.8 / 2:  # room 116 bottom
            self.dest = np.array([735 / 2 + buffer, 751.1 / 2 + buffer + (885.8 / 2) - 150 / 2 + 2 * 885.8 / 2])

        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + 297 / 2 < self.pos[1] < 751.1 / 2 + 885.8 / 2 + 885.8 /4 + buffer:  # 118-117 corridor
            self.dest = np.array([735 / 2 + 297/4 + buffer, 751.1 / 2 + 297/4 + buffer])


        #L2
        elif 735/2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 100 / 2 + buffer < self.pos[1] < 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 100 / 2 + buffer + 297/2:
            self.dest = np.array(exits_info[chosen_exit_index_L2][0])
            

        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + 885.8 / 2 + 885.8 / 4 + buffer < self.pos[1] < 751.1 / 2 + 3 * (885.8 / 2) + buffer + (3*297)/2:  # 116-117 corridor to gonz exit
            self.dest = np.array([735 / 2 + 297 / 4 + buffer, 751.1 / 2 + 3 * (885.8 / 2) + 297 / 2 + 734/2 + buffer + 500])
        elif 1032.1 / 2 + buffer < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 4 + buffer and 751.1 / 2 + 3 * (885.8 / 2) + buffer < self.pos[1] < 751.1 / 2 + 3 * (885.8 / 2) + buffer + 297 / 2:  # bottom corridor left
            self.dest = np.array([735 / 2 + 297 / 4 + buffer, 751.1 / 2 + 3 * (885.8 / 2) + 297 / 2 + buffer - 25])
        elif 1032.1 / 2 + buffer < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 2 + 2 * (284 / 2) + 437 / 2 + buffer + 479.5 / 2 and 751.1 / 2 + 3 * (885.8 / 2) + buffer < self.pos[1] < 751.1 / 2 + 3 * (885.8 / 2) + buffer + 297 / 2:  # bottom corridor right
            self.dest = np.array([1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 2 + 2 * (284 / 2) + 437 / 2 + buffer + 479.5 / 2 + 500,751.1 / 2 + 3 * (885.8 / 2) + 297 / 4 + buffer])



        elif 1032.1/2 + 324/2 + buffer < self.pos[0] < 1032.1/2 + 324/2 + 884/2 + buffer - 884/4 and 751.1/2 + (3*885.8)/2 + 297/2 + buffer < self.pos[1] < 751.1/2 + (3*885.8)/2 + 297/2 + 734/2 + buffer: #109 left
            self.dest = np.array([1032.1/2 + 324/2 + 100/2 + buffer + 50/2, 751.1/2 + (3*885.8)/2 + 297/2 + buffer])
        elif 1032.1/2 + 324/2 + 884/2 + buffer - 884/4 < self.pos[0] < 1032.1/2 + 324/2 + 884/2 + buffer and 751.1/2 + (3*885.8)/2 + 297/2 + buffer < self.pos[1] < 751.1/2 + (3*885.8)/2 + 297/2 + 734/2 + buffer: #109 right
            self.dest = np.array([1032.1/2 + 324/2 + 884/2 + buffer - 100/2 - 50/2, 751.1/2 + (3*885.8)/2 + 297/2 + buffer])

        elif 1032.1 / 2 + 324 / 2 + 884/2 + buffer < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884 / 2 + buffer + 1484 / 4 and 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + 734 / 2 + buffer:  # 105 left
            self.dest = np.array([1032.1 / 2 + 324 / 2 + 884/2 + buffer + 50/2, 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer])
        elif 1032.1 / 2 + 324 / 2 + 884 / 2 + buffer + 1484 / 4  < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 2 + buffer and 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + 734 / 2 + buffer:  # 105 right
            self.dest = np.array([1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 2 + buffer - 50/2, 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer])

        elif 1032.1 / 2 + 324 / 2 + 884/2 + 1484/2 + buffer < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884 / 2 + buffer + 1484 / 2 + 2*(284/2) and 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + 734 / 2 + buffer:  # reception
            self.dest = np.array([1032.1 / 2 + 324 / 2 + 884/2 + 1484/2 + buffer + 150/2, 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer])

        elif 1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 2 + 2*(284/2) + buffer < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884 / 2 + buffer + 1484 / 2 + 2 * (284 / 2) + 437/2 and 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + 734 / 2 + buffer:  # 102
            self.dest = np.array([1032.1 / 2 + 324 / 2 + 884 / 2 + 1484 / 2 + 2*(284/2) + 437/2 + buffer - 150/2,751.1 / 2 + (3 * 885.8) / 2 + 297 / 2 + buffer])

