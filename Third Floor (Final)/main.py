

import pygame

# Importing locals
from pygame.locals import *

from additional_functions import *
from agent import *
from environment import *


# Other packages
import sys
import numpy as np
import numpy.random as random
import math
import time


# Initializing Pygame and font
pygame.init()
pygame.font.init()
timefont = pygame.font.SysFont('Arial', 30)

""" 

Creating a screen with a room that is smaller than then screen 

"""

# Size of the screen
width = 1500
height = 800
size = width, height  # Do not adjust this

# Creating screen
roomscreen = pygame.display.set_mode(size)

# Making background white and creating colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLACK = (0, 0, 0)
background_color = WHITE
roomscreen.fill(background_color)
pygame.display.update()

# Defining clock
clock = pygame.time.Clock()

# Move world view
world_offset = [0, 0]

#################

# Function to initialize agents in a grid pattern within a room
def initialize_agents(room_dict, num_agents, agent_size):
    """Initializes agents in a grid within a room."""

    agents = []
    room_width = room_dict["topright"][0] - room_dict["bottomleft"][0]
    room_height = room_dict["topright"][1] - room_dict["bottomleft"][1]

    grid_width = 2  # Fixed grid width for 4x8 arrangement
    grid_height = 2  # Fixed grid height for 4x8 arrangement

    spacing_x = room_width / (grid_width)  # Calculate spacing for even distribution
    spacing_y = room_height / (grid_height)

    for row in range(grid_height):
        y_offset = room_dict["bottomleft"][1] + (row) * spacing_y
        for col in range(grid_width):
            x_offset = room_dict["bottomleft"][0] + (col) * spacing_x
            agent_x = x_offset  # Use x_offset directly for centered grid placement
            agent_y = y_offset  # Use y_offset directly for centered grid placement
            agents.append((agent_x, agent_y))

            if len(agents) == num_agents:  # Stop if enough agents are placed
                break

    return agents


buffer = 100
rooms = {
    "a321": {
        "bottomleft": (1032.1 / 2 + 3*buffer/2, 3*buffer/2),
        "topright": (1917.9 / 2 + 885.8 / 2 + 3*buffer/2, 751.1 / 2 + 297 / 2 + 3*buffer/2)
    },
    "a317": {
        "bottomleft": (3*buffer/2, 751.1 / 2 + 3*buffer/2),
        "topright": (735 / 2  + 3*buffer/2, 751.1 / 2 + 885.8 / 2  + 3*buffer/2)
    },
    "a313": {
        "bottomleft": (3*buffer/2, 751.1 / 2 + 885.8 / 2  + 3*buffer/2),
        "topright": (735 / 2  + 3*buffer/2, 751.1 / 2 + 885.8 / 2 + 855.6 / 2  + 3*buffer/2)
    },
    "a309": {
        "bottomleft": (3*buffer/2, 751.1 / 2 + 885.8 / 2 + 855.6 / 2  + 3*buffer/2),
        "topright": (735 / 2  + 3*buffer/2, 751.1 / 2 + (3 * 885.8) / 2  + 3*buffer/2)
    },
    "a304": {
        "bottomleft": (1032.1 / 2 + 3*buffer/2 + 3 * (284 / 2), 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2  + 3*buffer/2),
        "topright": (1032.1 / 2 + 3*buffer/2 + 3 * (284 / 2) + 937.3 / 2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2  + 3*buffer/2)
    },
    "a303": {
        "bottomleft": (1032.1 / 2 + 3*buffer/2 + 3 * (284 / 2) + 937.3 / 2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2  + 3*buffer/2),
        "topright": (1032.1 / 2 + 3*buffer/2 + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2  + 3*buffer/2)
    },
    "a302": {
        "bottomleft": (1032.1 / 2 + 3*buffer/2 + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2  + 3*buffer/2),
        "topright": (1032.1 / 2 + 3*buffer/2 + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2, 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 734 / 2  + 3*buffer/2)
    }
}

num_agents_per_group = 10
agent_size = np.random.normal(0.4335*(100/4), 0.0405*(100/4))

nr_rooms = len(rooms)
nr_agents = num_agents_per_group * nr_rooms

nr_experiments = 10

# Making sure we can run experiments one by one
j = 0  # add one after running

# Checking if all are executed
if j == nr_agents:
    print("nr of experiments reached")
    sys.exit()

data_matrix = np.zeros((nr_experiments * nr_agents, 7))  # Delete/comment after first run

COORD = []
# Initialize agents in each room
i = 0
for room_name, room_dict in rooms.items():
    agents = initialize_agents(room_dict, num_agents_per_group, agent_size)
    COORD.append(agents)
    i+=1
    
#Exit Choice Params
exits_info = [(np.array([735/2 + buffer, 751.1 / 2 + (3 * 885.8 / 2) + 289.4/2 + 297/2 + 289.4/4 + buffer]), 750), #bottom left
               (np.array([1032.1 / 2 + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 + buffer, 751.1 / 2 + (3 * 885.8 / 2) + 289.4 / 4 + buffer]), 750), #bottom right
               (np.array([buffer + 100 / 2 + 50 / 2, 751.1 / 2 + buffer - 241.8 / 2]), 1000)]  #fire exit
pmax = 1
k_d = 1.2 #exits significantly more probable
density_data_per_time = {}


#################

def main():
    # Now to let multiple objects move to the door we define
    # nr_agents = nr_agents
    agent_color = GREEN
    line_color = BLACK

    # initialize agents
    agents = []

    def positions(agents):
        for room in COORD:
            for x,y in room:
                agent = Agent(x,y)
                agent.walls = walls
                agent.radius = np.random.normal(0.4335*(100/4), 0.0405*(100/4))
                agent.mass = np.random.normal(54.7,11.4)
                agent.dSpeed = 0.75*50
                agents.append(agent)

    positions(agents)

    # Panic Force
    def apply_panic_force_percentage(agents, panic_multiplier=2, percentage=50):
        num_agents_to_affect = int(len(agents) * (percentage / 100))
        selected_agents = np.random.choice(agents, num_agents_to_affect, replace=False)
        for agent in selected_agents:
            agent.dSpeed *= panic_multiplier

    apply_panic_force_percentage(agents)

    def apply_herding_force(agents, herding_radius=50, herding_strength=2):
        for agent in agents:
            neighbors = [other for other in agents if
                         np.linalg.norm(agent.pos - other.pos) < herding_radius and other != agent]

            if neighbors:
                # Compute the average destination of neighbors
                avg_dest = np.mean([neighbor.pos for neighbor in neighbors], axis=0)

                # Adjust the agent's direction towards this average destination
                new_direction = normalize(avg_dest - agent.pos)
                agent.direction = new_direction * herding_strength
                # Optionally adjust velocity as well
                agent.aVelocity = normalize(agent.aVelocity + new_direction * herding_strength)

    apply_herding_force(agents)

    count = 0
    start_time = time.time()
    run = True

    while run:

        # Updating time
        if count < nr_agents:
            current_time = time.time()
            elapsed_time = current_time - start_time
        else:
            for agent_i in agents:
                data_matrix[(j + 1) * nr_agents - 2][0] = elapsed_time
                data_matrix[(j + 1) * nr_agents - 1][0] = elapsed_time
                agents.remove(agent_i)
            for k in range(j * nr_agents, (j + 1) * nr_agents):
                data_matrix[k][1] = elapsed_time

        # Finding delta t for this frame
        dt = clock.tick(70) / 1000

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                (x, y) = pygame.mouse.get_pos()
                print(x, y)

        global world_offset

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            world_offset[0] = world_offset[0] + 30
        if keys[pygame.K_RIGHT]:
            world_offset[0] = world_offset[0] - 30
        if keys[pygame.K_UP]:
            world_offset[1] = world_offset[1] + 30
        if keys[pygame.K_DOWN]:
            world_offset[1] = world_offset[1] - 30

        roomscreen.fill(background_color)

        # draw walls
        for wall in walls:
            start_posw = np.array([wall[0], wall[1]])
            end_posw = np.array([wall[2], wall[3]])
            start_posx = start_posw
            end_posx = end_posw
            pygame.draw.line(roomscreen, line_color, start_posx + world_offset, end_posx + world_offset, 3)

        for agent_i in agents:
            agent_i.update_dest(agents, exits_info, pmax, k_d)
            agent_i.direction = normalize(agent_i.dest - agent_i.pos)
            agent_i.dVelocity = agent_i.dSpeed * agent_i.direction
            aVelocity_force = agent_i.velocity_force()
            people_interaction = 0.0
            wall_interaction = 0.0
            
            _, densities = agent_i.calculate_exit_probabilities(agents, exits_info, pmax, k_d)
            current_time_step = agent_i.time
            
            if current_time_step not in density_data_per_time:
                density_data_per_time[current_time_step] = [densities]
            else:
                density_data_per_time[current_time_step].append(densities)

            for agent_j in agents:
                if agent_i == agent_j: continue
                people_interaction += agent_i.f_ij(agent_j)

            for wall in walls:
                wall_interaction += agent_i.f_ik_wall(wall)

            sumForce = aVelocity_force + people_interaction + wall_interaction + random.uniform(0, 1)
            dv_dt = sumForce / agent_i.mass
            agent_i.aVelocity = agent_i.aVelocity + dv_dt * dt
            agent_i.pos = agent_i.pos + agent_i.aVelocity * dt

        for agent_i in agents:

            agent_i.time += clock.get_time() / 1000
            start_position = [0, 0]
            start_position[0] = int(agent_i.pos[0]) + world_offset[0]
            start_position[1] = int(agent_i.pos[1]) + world_offset[1]

            end_position = [0, 0]
            end_position[0] = int(agent_i.pos[0] + agent_i.aVelocity[0]) + world_offset[0]
            end_position[1] = int(agent_i.pos[1] + agent_i.aVelocity[1]) + world_offset[1]

            end_positionDV = [0, 0]
            end_positionDV[0] = int(agent_i.pos[0] + agent_i.dVelocity[0]) + world_offset[0]
            end_positionDV[1] = int(agent_i.pos[1] + agent_i.dVelocity[1]) + world_offset[1]

            if ((start_position[0] <= 200/2 + buffer + world_offset[0] and start_position[1] <= 751.1/2 - 241.8/2 + buffer + world_offset[1]) or
                (start_position[0] <= 735/2 + buffer + world_offset[0] and start_position[1] >= 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 297/2 + buffer + world_offset[1]) or
                (start_position[0] >= 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 + world_offset[0])) and agent_i.Goal == 0:
                agent_i.Goal = 1
                data_matrix[count + j * nr_agents][0] = agent_i.time
                # print('Time to Reach the Goal:', agent_i.time)

            if ((start_position[0] <= 200/2 + buffer + world_offset[0] and start_position[1] <= 751.1/2 - 241.8/2 + buffer + world_offset[1]) or
                (start_position[0] <= 735/2 + buffer + world_offset[0] and start_position[1] >= 751.1 / 2 + (3 * 885.8) / 2 + 289.4 / 2 + 297/2 + buffer + world_offset[1]) or
                (start_position[0] >= 1032.1 / 2 + buffer + 3 * (284 / 2) + 937.3 / 2 + 1189.2 / 2 + 1184/2 + world_offset[0])):
                data_matrix[count + j * nr_agents][2] = count
                data_matrix[count + j * nr_agents][3] = agent_i.countcollision
                data_matrix[count + j * nr_agents][4] = np.linalg.norm(agent_i.aVelocity)
                data_matrix[count + j * nr_agents][5] = agent_i.dSpeed
                data_matrix[count + j * nr_agents][6] = agent_i.dSpeed - np.linalg.norm(agent_i.aVelocity)
                count += 1
                agents.remove(agent_i)

            pygame.draw.circle(roomscreen, agent_color, start_position, round(agent_i.radius), 3)
            pygame.draw.line(roomscreen, agent_color, start_position, end_positionDV, 2)
        # visibility
        # pygame.draw.circle(roomscreen, RED, [700,400], 300, 1)

        pygame.draw.line(roomscreen, [255, 60, 0], start_position, end_positionDV, 2)

        # Present text on screen
        timestr = "Time: " + str(elapsed_time)
        agentstr = "Number of Agents Escaped: " + str(count) + "/" + str(nr_agents)
        timesurface = timefont.render(timestr, False, (0, 0, 0))
        agentsurface = timefont.render(agentstr, False, (0, 0, 0))
        roomscreen.blit(timesurface, (0, 0))
        roomscreen.blit(agentsurface, (0, 40))
        # Update the screen
        pygame.display.flip()

    pygame.quit()


main()
#np.savetxt('room2_vis', data_matrix)

density_data_list = []
for time_step, densities in density_data_per_time.items():
    for density in densities:
        # Assuming 'density' is itself a list or tuple of density values per exit
        density_data_list.append([time_step] + density)
        
density_data_array= np.array(density_data_list)

# Save to a text file
np.savetxt("density_data.txt", density_data_array)