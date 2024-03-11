# Programming evacuation systems using Crowd Simulation
from matplotlib import pyplot as plt

# Loading the pygame package
import pygame

# Importing locals
from pygame.locals import *
from additional_functions import *
from environment import *
from agent import *

# Other packages
import sys
import numpy as np
import numpy.random as random
import math
import time


# data_matrix = np.loadtxt('room1_r', dtype=float) # Enable after first run
data_matrix = np.zeros((nr_experiments * nr_agents, 4))  # Delete/comment after first run

# Making sure we can run experiments one by one
j = 0  # add one after running

# Checking if all are executed
if j == nr_agents:
    print("nr of experiments reached")
    sys.exit()

# Initializing Pygame and font
pygame.init()
pygame.font.init()
timefont = pygame.font.SysFont('Comic Sans MS', 30)

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
world_offset = [0,0]

exit_times = []  # Initialize this for exit times


def main():
    # Now to let multiple objects move to the door we define
    # nr_agents = nr_agents
    agent_color = GREEN
    line_color = BLACK

    # initialize agents
    agents = []

    def positions(agents):
        for i in range(nr_agents):
            agent = Agent()
            agent.walls = walls
            agent.x = positionmatrix[j * nr_agents + i][0]
            agent.y = positionmatrix[j * nr_agents + i][1]
            agent.pos = np.array([agent.x, agent.y])
            agent.radius = positionmatrix[j * nr_agents + i][2]
            agent.mass = positionmatrix[j * nr_agents + i][3]
            agent.dSpeed = positionmatrix[j * nr_agents + i][4]
            agents.append(agent)

    positions(agents)
    
    #Panic Force
    def apply_panic_force_percentage(agents, panic_multiplier=2, percentage=50):
        num_agents_to_affect = int(len(agents) * (percentage / 100))
        selected_agents = np.random.choice(agents, num_agents_to_affect, replace=False)
        for agent in selected_agents:
            agent.dSpeed *= panic_multiplier
            
    apply_panic_force_percentage(agents)
    
    def apply_herding_force(agents, herding_radius=50, herding_strength=2):
        for agent in agents:
            neighbors = [other for other in agents if np.linalg.norm(agent.pos - other.pos) < herding_radius and other != agent]
            
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
            agent_i.update_dest()
            agent_i.direction = normalize(agent_i.dest - agent_i.pos)
            agent_i.dVelocity = agent_i.dSpeed * agent_i.direction
            aVelocity_force = agent_i.velocity_force()
            people_interaction = 0.0
            wall_interaction = 0.0

            for agent_j in agents:
                if agent_i == agent_j: continue
                people_interaction += agent_i.f_ij(agent_j)

            for wall in walls:
                wall_interaction += agent_i.f_ik_wall(wall)

            sumForce = aVelocity_force + people_interaction + wall_interaction
            dv_dt = sumForce / agent_i.mass
            agent_i.aVelocity = agent_i.aVelocity + dv_dt * dt
            agent_i.pos = agent_i.pos + agent_i.aVelocity * dt

            # Avoiding disappearing agents
            if agent_i.pos[0] > 2900 or agent_i.pos[0] < 50 or agent_i.pos[1] > 2300 or agent_i.pos[1] < 50:
                main()
                sys.exit()

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

            if ((start_position[0] >= 1156.3/2 + 4203/2 + buffer + world_offset[0]) or (start_position[1] >= 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer + world_offset[1] and start_position[0] <= buffer + 735/2 + 324/2 + world_offset[0]) ) and agent_i.Goal == 0:
                agent_i.Goal = 1
                data_matrix[count + j * nr_agents][0] = agent_i.time
                # print('Time to Reach the Goal:', agent_i.time)

            if (start_position[0] >= 1156.3/2 + 4203/2 + buffer + world_offset[0]) or (start_position[1] >= 751.1/2 + (3 * 885.8/2) + 321.6/2 + buffer + world_offset[1] and start_position[0] <= buffer + 735/2 + 324/2 + world_offset[0]):
                data_matrix[count + j * nr_agents][2] = count
                data_matrix[count + j * nr_agents][3] = agent_i.countcollision
                count += 1
                agents.remove(agent_i)
                agent_i.Goal = 1
                data_matrix[count + j * nr_agents][0] = agent_i.time
                exit_times.append(agent_i.time)  # Append the exit time here

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
    
    plt.figure(figsize=(10, 6))  
    plt.plot(exit_times, marker='o', linestyle='-', color='b') 
    plt.title('Exit Times of Agents')
    plt.xlabel('Agent Index')
    plt.ylabel('Exit Time (s)')
    plt.grid(True)  
    plt.show()



main()
# np.savetxt('room1_vis', data_matrix)
