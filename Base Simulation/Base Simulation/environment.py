# Importing packages
from additional_functions import *
import numpy as np
import math
import numpy.random as random

# Setting seed
random.seed(123)

# Creating a dataset
nr_agents = 54
nr_experiments = 10

# Walls list to check
room_height = 600 # height of the room
room_width = 600 # width of the room
room_left = 100 # left pixels coordinate
room_top = 100 # top pixels coordeinate

buffer = 100

# Door 1
door_ytop = 382 # 376
door_ybottom = 418 # 424
door_xleft = 382
door_xright = 418
second_room_offset = 600
corridor = 150

""" 
    Now we need to create the doors through which objects will leave in case of evacuation
    This door's position can be determined using:
"""

# This gives the following walls

#walls
walls = [ [1156.3/2 + buffer, buffer, 2042.1/2 + buffer, buffer], #A214
          [1156.3/2 + buffer, buffer, 1156.3/2 + buffer, 751.1/2 + buffer],
          [1156.3/2 + buffer, 751.1/2 + buffer, 1156.3/2 + 200/2 + buffer, 751.1/2 + buffer],
          [1156.3/2 + 200/2 + 100/2 + buffer, 751.1/2 + buffer, 1156.3/2 + 200/2 + 100/2 + 300/2 + buffer, 751.1/2 + buffer],
          [1156.3/2 + 200/2 + 100/2 + 300/2 + 100/2 + buffer, 751.1/2 + buffer, 1156.3/2 + 200/2 + 100/2 + 300/2 + 100/2 + 185.8/2 + buffer, 751.1/2 + buffer],
          [2042.1/2 + buffer, buffer, 2042.1/2 + buffer, 751.1/2 + buffer],

          [1156.3/2 + 885.8/2 + buffer, buffer, 2042.1/2 + 885.8/2 + buffer, buffer], #A215
          [1156.3/2 + 885.8/2 + buffer, buffer, 1156.3/2 + 885.8/2 + buffer, 751.1/2 + buffer],
          [1156.3/2 + 885.8/2 + buffer, 751.1/2 + buffer, 1156.3/2 + 885.8/2 + 185.8/2 + buffer, 751.1/2 + buffer],
          [1156.3/2 + 885.8/2 + 185.8/2 + 100/2 + buffer, 751.1/2 + buffer, 1156.3/2 + 885.8/2 + 185.8/2 + 100/2 + 300/2 + buffer, 751.1/2 + buffer],
          [1156.3/2 + 885.8/2 + 185.8/2 + 100/2 + 300/2 + 100/2 + buffer, 751.1/2 + buffer, 1156.3/2 + 885.8/2 + 185.8/2 + 100/2 + 300/2 + 100/2 + 200/2 + buffer,751.1/2 + buffer],
          [2042.1/2 + 885.8/2 + buffer, buffer, 2042.1/2 + 885.8/2 + buffer, 751.1/2 + 297/2 + buffer],

          [1156.3/2 + buffer, 751.1/2 + 297/2 + buffer, 2042.1/2 + 885.8/2 + buffer, 751.1/2 + 297/2 + buffer],

          [buffer, 751.1/2 + buffer, 735/2 + 421.3/2 + buffer, 751.1/2 + buffer], #210, 209, 208
          [buffer, 751.1/2 + buffer, buffer, 751.1/2 + (3*885.8)/2 + buffer],
          [735/2 + buffer, 751.1/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + buffer],
          [735/2 + buffer, 751.1/2 + 100/2 + 100/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + buffer],
          [735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + buffer],
          [735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + buffer],
          [735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + buffer],
          [735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + buffer],
          [735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + buffer, 735/2 + buffer, 751.1/2 + 100/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + 200/2 + 100/2 + 485.8/2 + 100/2 + 100/2 + 321.6/2 + buffer],
          [buffer, 751.1/2 + 885.8/2 + buffer, 735/2 + buffer, 751.1/2 + 885.8/2 + buffer],
          [buffer, 751.1/2 + (2*885.8)/2 + buffer, 735/2 + buffer, 751.1/2 + + (2*885.8)/2 + buffer],
          [buffer, 751.1/2 + (3*885.8)/2 + buffer, 735/2 + buffer, 751.1/2 + + (3*885.8)/2 + buffer],

          [1156.3/2 + buffer, 751.1/2 + 297/2 + buffer, 1156.3/2 + buffer, 751.1/2 + (3*885.8)/2 + buffer],
          [1156.3/2 + buffer, 751.1/2 + (3*885.8)/2 + buffer, 1156.3/2 + 4203/2 + buffer, 751.1/2 + (3*885.8)/2 + buffer],

          [1156.3/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer, 1156.3/2 + 4203/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + 324/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 324/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + 324/2 + 892/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 324/2 + 892/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + 324/2 + (2*892)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 324/2 + (2*892)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + 324/2 + (2*892)/2 + 884/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 324/2 + (2*892)/2 + 884/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + 324/2 + (2*892)/2 + (2*884/2) + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 324/2 + (2*892)/2 + (2*884)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],
          [1156.3/2 + 324/2 + (2*892)/2 + (2*884/2) + 327/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 324/2 + (2*892)/2 + (2*884)/2 + 327/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + 734/2 + buffer],

          [1156.3/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + 100/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + 100/2 + 492/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (2*100)/2 + 492/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + (4*100)/2 + 492/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (5*100)/2 + 492/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + (5*100)/2 + (2*492)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (6*100)/2 + (2*492)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + (8*100)/2 + (2*492)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (9*100)/2 + (2*492)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer,  1156.3/2 + 424/2 + (9*100)/2 + (2*492)/2 + 484/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (10*100)/2 + (2*492)/2 + 484/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + (12*100)/2 + (2*492)/2 + 484/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (13*100)/2 + (2*492)/2 + 484/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + (13*100)/2 + (2*492)/2 + (2*484)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer],
          [1156.3/2 + 424/2 + (14*100)/2 + (2*492)/2 + (2*484)/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer, 1156.3/2 + 424/2 + (15*100)/2 + (2*492)/2 + (2*484)/2 + 327/2 + buffer, 751.1/2 + (3*885.8)/2 + 321.6/2 + buffer]

             ]

# List to save positions
positionmatrix = []
# For all experiments
for j in range(0, nr_experiments):
    nr_experiment = j + 1
    agents_found = 0
    # room 1
    for i in range(0, nr_agents):  # For all objects
        # We start by finding a random position in the room
        found = False
        countwall = 0
        while found == False:
            countwall = 0
            desiredS = np.random.uniform(20,30)
            mass = np.random.uniform(60,100)
            radius = 12 / 80 * mass

            if agents_found < int(nr_agents / 9):
                object_x = np.random.uniform(1156.3/2 + 885.8/2 + buffer, 2042.1/2 + 885.8/2 + buffer)
                object_y = np.random.uniform(buffer, 751.1/2 + buffer)
            elif agents_found < int(2 * nr_agents / 9):
                object_x = np.random.uniform(1156.3/2 + buffer, 2042.1/2 + buffer)
                object_y = np.random.uniform(buffer, 751.1/2 + buffer)
            elif agents_found < int(3 * (nr_agents / 9)):
                object_x = np.random.uniform(buffer, 735/2 + buffer)
                object_y = np.random.uniform(751.1/2 + buffer, 751.1/2 + 885.8/2 + buffer)
            elif agents_found < int(4 * (nr_agents / 9)):
                object_x = np.random.uniform(buffer, 735 / 2 + buffer)
                object_y = np.random.uniform(751.1/2 + 885.8/2 + buffer, 751.1/2 + (2*885.8/2) + buffer)
            elif agents_found < int(5 * (nr_agents / 9)):
                object_x = np.random.uniform(buffer, 735 / 2 + buffer)
                object_y = np.random.uniform(751.1/2 + (2*885.8/2) + buffer, 751.1/2 + (3 * 885.8/2) + buffer)
            elif agents_found < int(6 * (nr_agents / 9)):
                object_x = np.random.uniform(1156.3/2 + 324/2 + buffer, 1156.3/2 + 324/2 + 892/2 + buffer)
                object_y = np.random.uniform(751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer, 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer)
            elif agents_found < int(7 * (nr_agents / 9)):
                object_x = np.random.uniform(1156.3/2 + 324/2 + 892/2 + buffer, 1156.3/2 + 324/2 + (2*892/2) + buffer)
                object_y = np.random.uniform(751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer, 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer)
            elif agents_found < int(8 * (nr_agents / 9)):
                object_x = np.random.uniform(1156.3/2 + 324/2 + (2*892/2) + buffer, 1156.3/2 + 324/2 + (2*892/2) + 884/2 + buffer)
                object_y = np.random.uniform(751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer, 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer)
            else:
                object_x = np.random.uniform(1156.3/2 + 324/2 + (2*892/2) + 884/2 + buffer, 1156.3/2 + 324/2 + (2*892/2) + (2*884/2) + buffer)
                object_y = np.random.uniform(751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer, 751.1/2 + (3 * 885.8/2) + 321.6/2 + 734/2 + buffer)

            for wall in walls:
                r_i = radius
                d_iw, e_iw = distance_agent_to_wall(np.array([object_x, object_y]), wall)
                if d_iw < r_i:
                    countwall += 1

            if len([positionmatrix[i] for i in range(j * nr_agents, j * nr_agents + agents_found)]) > 0:
                countagents = 0
                for position in [positionmatrix[i] for i in range(j * nr_agents, j * nr_agents + agents_found)]:
                    dist = math.sqrt((position[0] - object_x) ** 2 + (position[1] - object_y) ** 2)
                    if dist > position[2] + radius:
                        countagents += 1
                if countagents == i and countwall == 0:
                    found = True
                    agents_found += 1
            elif countwall == 0:
                found = True
                agents_found += 1
        positionmatrix.append([object_x, object_y, radius, mass, desiredS, nr_experiment])




