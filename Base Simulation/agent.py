import numpy as np
import numpy.random as random

# Importing locals
from pygame.locals import *
from additional_functions import *
from environment import *

# Creating evacuation object class
class Agent(object):
    def __init__(self):

        self.mass = 80  # random.uniform(40,90)
        self.radius = 20
        # random initialize a agent

        self.x = random.uniform(100 + self.radius, 600 - self.radius + 600)
        self.y = random.uniform(100 + self.radius, 1300 - self.radius + 600)
        self.pos = np.array([self.x, self.y])
        # self.pos = np.array([10.0, 10.0])
        
        # Initialize dx and dy for herding
        self.dx = 0
        self.dy = 0
        
        self.aVelocityX = random.uniform(0,16)
        self.aVelocityY = random.uniform(0,16)
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

        print('X and Y Position:', self.pos)
        print('self.direction:', self.direction)

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
    

    def update_dest(self):
        #dist = math.sqrt((self.pos[0] - 700) ** 2 + (self.pos[1] - 400) ** 2)
        if  1156.3/2 + 885.8/2 + buffer < self.pos[0] < 1156.3/2 + 885.8/2 + (885.8/2)/2 + buffer and buffer < self.pos[1] < 751.1/2 + buffer: #room 215 left
            self.dest = np.array([1156.3/2 + 885.8/2 + buffer + 235.8/2, 751.1/2 + buffer + 50])
        elif  (1156.3/2 + 885.8/2 + (885.8/2)/2 + buffer) < self.pos[0] < 1156.3/2 + 885.8/2 + (885.8/2) + buffer and buffer < self.pos[1] < 751.1/2 + buffer: #room 215 right
            self.dest = np.array([1156.3/2 + 885.8/2 + (885.8/2) + buffer - 250/2, 751.1/2 + buffer + 50])
        elif 1156.3 / 2 + buffer < self.pos[0] < 1156.3 / 2 + (885.8 / 2) / 2 + buffer and buffer < self.pos[1] < 751.1 / 2 + buffer:  # room 214 left
            self.dest = np.array([1156.3 / 2 + buffer + 250 / 2, 751.1 / 2 + buffer + 50])
        elif (1156.3 / 2 + (885.8 / 2) / 2 + buffer) < self.pos[0] < 1156.3 / 2 + (885.8 / 2) + buffer and buffer < self.pos[1] < 751.1 / 2 + buffer:  # room 214 right
            self.dest = np.array([1156.3 / 2 + (885.8 / 2) + buffer - 235.8 / 2, 751.1 / 2 + buffer + 50])

        elif 1156.3 / 2 + buffer < self.pos[0] < 1156.3/2 + 885.8/2 + (885.8/2) + buffer and 751.1/2 + buffer < self.pos[1] < 751.1/2 + 297/2 + buffer: #corridor
            self.dest = np.array([1156.3 / 2 + buffer - 210.6/2,  751.1/2 + (297/2)/2 + buffer])
        elif 735 / 2 + buffer < self.pos[0] < 1156.3 / 2 + buffer and 751.1 / 2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer: #corridor
            self.dest = np.array([1156.3 / 2 + buffer - 210.6/2,  751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer + 50])

        elif buffer < self.pos[0] < 735/2 + buffer and 751.1/2 + buffer < self.pos[1] < 751.1/2 + buffer + (885.8/2)/2: #room 209 top
            self.dest = np.array([735/2 + buffer and 751.1/2 + buffer + 10, 751.1/2 + buffer + 150/2])
        elif buffer < self.pos[0] < 735/2 + buffer and 751.1/2 + buffer + (885.8/2)/2 < self.pos[1] < 751.1/2 + buffer + (885.8/2): #room 209 top
            self.dest = np.array([735/2 + buffer and 751.1/2 + buffer + 10, 751.1/2 + buffer + (885.8/2) - 150/2])

        elif buffer < self.pos[0] < 735/2 + buffer and 751.1/2 + buffer + 885.8/2 < self.pos[1] < 751.1/2 + buffer + (885.8/2)/2 + 885.8/2 : #room 209 top
            self.dest = np.array([735/2 + buffer and 751.1/2 + buffer + 10, 751.1/2 + buffer + 150/2 + 885.8/2 ])
        elif buffer < self.pos[0] < 735/2 + buffer and 751.1/2 + buffer + (885.8/2)/2 + 885.8/2 < self.pos[1] < 751.1/2 + buffer + (885.8/2) + 885.8/2 : #room 209 top
            self.dest = np.array([735/2 + buffer and 751.1/2 + buffer + 10, 751.1/2 + buffer + (885.8/2) - 150/2 + 885.8/2 ])

        elif buffer < self.pos[0] < 735/2 + buffer and 751.1/2 + buffer + 2*885.8/2 < self.pos[1] < 751.1/2 + buffer + (885.8/2)/2 + 2*885.8/2 : #room 209 top
            self.dest = np.array([735/2 + buffer and 751.1/2 + buffer + 10, 751.1/2 + buffer + 150/2 + 2*885.8/2 ])
        elif buffer < self.pos[0] < 735/2 + buffer and 751.1/2 + buffer + (885.8/2)/2 + 2*885.8/2 < self.pos[1] < 751.1/2 + buffer + (885.8/2) + 2*885.8/2 : #room 209 top
            self.dest = np.array([735/2 + buffer and 751.1/2 + buffer + 10, 751.1/2 + buffer + (885.8/2) - 150/2 + 2*885.8/2 ])

        elif 1156.3 / 2 + buffer < self.pos[0] < 1156.3/2 + 324/2 + 8920 + 8840 + 3270 + buffer and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer - 321.6/2 < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer: #corridor
            self.dest = np.array([1156.3/2 + 4203/2 + buffer,  751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer - (321.6/2)/2])

        elif 1156.3/2 + 324/2 + buffer < self.pos[0] < 1156.3/2 + 324/2 + buffer + (892/2)/2 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 150/2, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])
        elif 1156.3/2 + 324/2 + buffer + (892/2)/2 < self.pos[0] < 1156.3/2 + 324/2 + buffer + 892/2 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 892/2 - 150/2, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])

        elif 1156.3/2 + 324/2 + buffer + 892/2< self.pos[0] < 1156.3/2 + 324/2 + buffer + (892/2)/2  + 892/2and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 150/2 + 892/2, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])
        elif 1156.3/2 + 324/2 + buffer + (892/2)/2 + 892/2< self.pos[0] < 1156.3/2 + 324/2 + buffer + 892/2 + 892/2 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 892/2 - 150/2 + 892/2, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])

        elif 1156.3/2 + 324/2 + buffer + 892/2 + 884/2 < self.pos[0] < 1156.3/2 + 324/2 + buffer + (892/2)/2  + 892/2 + 884/2 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 150/2 + 892/2 + 884/2, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])
        elif 1156.3/2 + 324/2 + buffer + (892/2)/2 + 892/2 + 884/2 < self.pos[0] < 1156.3/2 + 324/2 + buffer + 892/2 + 892/2 + 884/2 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 892/2 - 150/2 + 892/2 + 884/2, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])

        elif 1156.3/2 + 324/2 + buffer + 892/2 + 884 < self.pos[0] < 1156.3/2 + 324/2 + buffer + (892/2)/2  + 892/2 + 884 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 150/2 + 892/2 + 884, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])
        elif 1156.3/2 + 324/2 + buffer + (892/2)/2 + 892/2 + 884 < self.pos[0] < 1156.3/2 + 324/2 + buffer + 892/2 + 892/2 + 884 and 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer < self.pos[1] < 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer: #room 208
            self.dest = np.array([1156.3/2 + 324/2 + buffer + 892/2 - 150/2 + 892/2 + 884, 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer])




            #self.dest = normalize(self.dest-self.pos) * 1000 + self.dest
            #if self.pos[1] > 690 - self.radius:
            #    self.dest = np.array([700,self.pos[1]]) # turn left
            #elif self.pos[1] < 110 + self.radius:
            #    self.dest = np.array([700,self.pos[1]]) # turn right
            #elif self.pos[0] < 110 + self.radius and self.pos[1] < 400:
            #    self.dest = np.array([self.pos[0], 100]) # turn right
            #elif self.pos[0] < 110 + self.radius:
            #    self.dest = np.array([self.pos[0], 700]) # turn left"""