import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt

# Importing locals
from pygame.locals import *
from additional_functions import *
from environment import *

acclTime_mult = 1
bodyFactor_mult = 1
F_mult = 1
delta_mult = 0.75


# Creating evacuation object class
class Agent(object):
    def __init__(self, x, y):

        self.mass = 80  # random.uniform(40,90)
        self.radius = 20
        # random initialize a agent

        # self.x = random.uniform(100 + self.radius, 600 - self.radius + 600)
        # self.y = random.uniform(100 + self.radius, 1300 - self.radius + 600)
        # print("heyhyeyehe")
        # print(self.x, self.y)
        # print("krjhahjd")
        self.x  = x
        self.y = y
        self.pos = np.array([self.x, self.y])
        # self.isInExit1=False
        # self.isInExit2=False
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

        self.acclTime = 0.5 * acclTime_mult # random.uniform(8,16) #10.0
        self.drivenAcc = (self.dVelocity - self.aVelocity) / self.acclTime

        self.bodyFactor = 240000 * bodyFactor_mult #kappa
        self.F = 2000 * F_mult
        self.delta = 0.08 * 50 * delta_mult # random.uniform(0.8,1.6) #0.8 #0.08

        self.Goal = 0
        self.time = 0.0
        self.countcollision = 0

        print('X and Y Position:', self.pos)
        print('self.direction:', self.direction)
        print('self.acclTime:', self.acclTime)

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
        # check if inside hardcoded semicircle exit 1. self.isInExit1 =True
        # exit 2

    def update_dest(self):
        # dist = math.sqrt((self.pos[0] - 700) ** 2 + (self.pos[1] - 400) ** 2)

        '''if 1032.1/2 + buffer < self.pos[0] < 1917.9/2 + buffer - 885.8/4 and buffer < self.pos[1] < 751.1/2 + buffer: #123left
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
            self.dest = np.array([1032.1 / 2 + 885.8 / 2 + (885.8 / 2) + buffer - 297/4, 751.1 / 2 + (297 / 2) + buffer + 500/2])'''

        if buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1 / 2 + buffer + ( 885.8 / 2) / 2:  # room 118 top
            self.dest = np.array([735 / 2 + buffer + 10, 751.1 / 2 + buffer + 150 / 2])
        elif buffer < self.pos[0] < 735 / 2 + buffer and 751.1 / 2 + buffer + (885.8 / 2) / 2 < self.pos[1] < 751.1 / 2 + buffer + (885.8 / 2):  # room 118 bottom
            self.dest = np.array( [735 / 2 + buffer + 10, 751.1 / 2 + buffer + (885.8 / 2) - 150 / 2])
        '''
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
        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + buffer and 751.1 / 2 + 885.8 / 2 + 885.8 /4 + buffer < self.pos[1] < 751.1 / 2 + 3 * (885.8 / 2) + buffer:  # 116-117 corridor
            self.dest = np.array([735 / 2 + 297/4 + buffer, 751.1 / 2 + 3 * (885.8 / 2) + 297/4 + buffer])
        elif 735 / 2 + buffer < self.pos[0] < 1032.1 / 2 + 324 / 2 + 884/2 + 1484/2 + 2*(284/2) + 437/2 + buffer + 479.5/2 and 751.1 / 2 + 3 * (885.8 / 2) + buffer < self.pos[1] < 751.1 / 2 + 3 * (885.8 / 2) + buffer + 297/2:  # bottom corridor
            self.dest = np.array([1032.1 / 2 + 324 / 2 + 884/2 + 1484/2 + 2*(284/2) + 437/2 + buffer + 479.5/2 + 500/2, 751.1 / 2 + 3 * (885.8 / 2) + 297/4 + buffer])

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
'''
