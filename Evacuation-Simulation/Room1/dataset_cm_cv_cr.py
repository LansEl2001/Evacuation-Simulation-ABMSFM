# Importing packages
from additional_functions import *
import numpy as np
import math
import numpy.random as random
# Setting seed
random.seed(123)
# Creating a dataset
nr_agents = 30
nr_experiments = 10


# Walls list to check
room_height = 600 # height of the room
room_width = 600 # width of the room
room_left = 100 # left pixels coordinate
room_top = 100 # top pixels coordeinate

# Door 1
door_ytop = 385
door_ybottom = 415
    
# Door 2
door_ytop = 385
door_ybottom = 415

walls = [[room_left, room_top, room_left + room_width, room_top], 
    [room_left, room_top, room_left, door_ytop], 
    [room_left, room_top+room_height, room_left, door_ybottom],
    [655,375,655,425], [655,375,670,400], [670,400,655,425], # additional walls
    [140, 375, 140, 425], [140, 375, 125, 400], [140, 425, 125, 400],  # additional walls
    [room_left, room_top+room_height, room_left + room_width, room_top+ room_height],
    [room_left + room_width, room_top, room_left + room_width, door_ytop],
    [room_left+room_width, room_top + room_height, room_left + room_width, door_ybottom]]

# List to save positions
positionmatrix = []


for j in range(0, nr_experiments):
    nr_experiment = j + 1
    agents_found = 0

    # Grid parameters
    grid_rows = 5
    grid_columns = 6
    grid_spacing = 80  # Spacing between grid positions

    # Calculate starting position of the grid
    grid_start_x = room_left + (room_width - (grid_columns - 1) * grid_spacing) / 2
    grid_start_y = room_top + (room_height - (grid_rows - 1) * grid_spacing) / 2

    for row in range(grid_rows):
        for column in range(grid_columns):
            # Calculate the position of the agent in the grid
            object_x = grid_start_x + column * grid_spacing
            object_y = grid_start_y + row * grid_spacing

            # Check if the position intersects with any walls
            countwall = 0
            for wall in walls:
                radius = 12 / 80 * 80  # Assuming mass = 80 for all agents
                r_i = radius
                d_iw, e_iw = distance_agent_to_wall(np.array([object_x, object_y]), wall)
                if d_iw < r_i:
                    countwall += 1

            if countwall == 0:
                # Add the agent's position to the position matrix
                positionmatrix.append([object_x, object_y, radius, 80, 20, nr_experiment])
                agents_found += 1

                # Break the loop if all agents are found
                if agents_found == nr_agents:
                    break

        # Break the loop if all agents are found
        if agents_found == nr_agents:
            break
