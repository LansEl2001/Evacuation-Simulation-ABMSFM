
import numpy as np
import numpy.random as random

# Importing locals
from pygame.locals import *
from additional_functions import *
from environment import *

# Creating evacuation object class
class Agent(object):
    def __init__(self,x,y):

        self.mass = np.random.normal(54.7,11.4)
        self.radius = np.random.normal(0.4335*(100/4), 0.0405*(100/4))
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

        self.dSpeed = 0.75*50
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
        print(f"Dens: {densities}")
        normalized_probabilities = [p / total if total > 0 else 0 for p in probabilities]
        #print(f"NProb: {normalized_probabilities}")
        
        return normalized_probabilities, densities
    
    #Position

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
        
        if 1032.1 / 2 + buffer < self.pos[0] < 1917.9 / 2 + 885.8 / 2 + buffer and buffer < self.pos[1] < 751.1 / 2 + 297 / 2 + buffer:  # facilty room
            self.dest = np.array([1032.1 / 2 + buffer, 751.1 / 2 + buffer - 241.8 / 4])
            
            
        #L1
        elif 735/2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 - 50 < self.pos[1] < 751.1 / 2 + 297/2 - 50:
            self.dest = np.array(exits_info[chosen_exit_index_L1][0])
            
            

        elif buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + buffer - 241.8 / 2 < self.pos[1] < 751.1 / 2 + buffer:  # fire exit corridor
            self.dest = np.array([buffer + 100 / 2 + 50 / 2, 751.1 / 2 + buffer - 241.8 / 2])
            
        

        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1 / 2 + (1.5 * 885.8 / 2) + buffer:  # 317 corridor
            self.dest = np.array([735/2 + 297/4 + buffer, 751.1 / 2 - 241.8/4 + buffer])
        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + (1.5 * 885.8 / 2) + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8 / 2) + 289.4/2 + 297/2 + buffer:  # 309 corridor
            self.dest = np.array([735/2 + 297/4 + buffer, 751.1 / 2 + (3 * 885.8 / 2) + 289.4/2 + 297/2 + 289.4/4 + buffer])
            
            
            
        #L2
        elif 735/2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 100 / 2 + buffer < self.pos[1] < 751.1 / 2 + 100 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 200 / 2 + 100 / 2 + 485.8 / 2 + 100 / 2 + 100 / 2 + buffer + 297/2:
            self.dest = np.array(exits_info[chosen_exit_index_L2][0])   
            
            
            
        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + (3 * 885.8 / 2) + 289.4/2 + 297/2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8 / 2) + 289.4/2 + 297/2 + 289.4/2 + buffer:  # exit box bottom left
            self.dest = np.array([735/2 + buffer, 751.1 / 2 + (3 * 885.8 / 2) + 289.4/2 + 297/2 + 289.4/4 + buffer])

        elif buffer < self.pos[0] < 735/2 + buffer and 751.1 / 2 + buffer < self.pos[1] <  751.1 / 2 + 885.8 / 2  + buffer:  # 317
            self.dest = np.array([735 / 2 + buffer, 751.1 / 2 + 885.8 / 2 + buffer - 150/2])

        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + 885.8 / 2 + buffer < self.pos[1] < 751.1 / 2 + 885.8 / 2 + 855.6/2 + buffer:  # 313
            self.dest = np.array([735 / 2 + buffer, 751.1 / 2 + 885.8 / 2 + buffer + 200/2])

        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + 885.8 / 2 + 855.6 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + buffer:  # 313
            self.dest = np.array([735 / 2 + buffer, 751.1 / 2 + 885.8 / 2 + 855.6 / 2 + buffer + 332.8 / 2 + 50/2])

        elif 1032.1 / 2 + buffer + 3 * (284 / 2) < self.pos[0] < 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3/2 and 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2  + buffer:  # 304
            self.dest = np.array([1032.1 / 2 + buffer + 3 * (284 / 2) + 150/2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer])

        elif 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3/2 < self.pos[0] < 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2/4 and 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2 + buffer:  # 303 left
            self.dest = np.array([1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3/2 + 200/2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer])
        elif 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2/4 < self.pos[0] < 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2/2 and 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer <self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2 + buffer:  # 303 right
            self.dest = np.array([1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2/2 - 200/2,751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer])

        elif 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2/2 < self.pos[0] < 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/4 and 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer <self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2 + buffer:  # 302 left
            self.dest = np.array([1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2/2 + 200/2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer])
        elif 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/4 < self.pos[0] < 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 and 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2 + buffer:  # 302 right
            self.dest = np.array([1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 - 200/2,751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + buffer])

        elif 1032.1 / 2 + buffer < self.pos[0] < 1032.1 / 2 + 3*(284/2) + 937.3/2 + buffer and 751.1 / 2 + (3 * 885.8 / 2) + buffer <self.pos[1] < 751.1 / 2 + (3 * 885.8 / 2) + 289.4 / 2 + buffer:  # 304 corridor
            self.dest = np.array([735 / 2 + 297 / 4 + buffer, 751.1 / 2 + (3 * 885.8 / 2) + 289.4 / 4 + buffer])
        elif 1032.1 / 2 + 3*(284/2) + 937.3/2 + buffer < self.pos[0] < 1032.1 / 2 + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 + buffer and 751.1 / 2 + (3 * 885.8 / 2) + buffer < self.pos[1] < 751.1 / 2 + (3 * 885.8 / 2) + 289.4 / 2 + buffer:  # 303-302 corridor
            self.dest = np.array([1032.1 / 2 + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 + buffer, 751.1 / 2 + (3 * 885.8 / 2) + 289.4 / 4 + buffer])

