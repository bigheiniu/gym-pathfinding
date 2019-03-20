from .games.astar import astar
from .envs.partially_observable_env import partial_grid
from .games.gridworld import generate_grid, MOUVEMENT

import operator
import numpy as np
# reversed MOUVEMENT dict

ACTION = {mouvement: action for action, mouvement in dict(enumerate(MOUVEMENT)).items()}


def compute_action_planning(grid, start, goal):
    path = astar(grid, start, goal)

    action_planning = []
    for i in range(len(path) - 1):
        pos = path[i]
        next_pos = path[i + 1]

        # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
        mouvement = tuple(map(operator.sub, next_pos, pos))

        action_planning.append(ACTION[mouvement])

    return path, action_planning

GOAL_VALUE = 3

'''
#Attention:
    Add sign to the visualization.
    5 -> turn left on your current direction in the next crossroad (left -> down, down -> right, right -> up, up -> left)
    6 -> turn right on your current direction in next crossroad (left -> up, up -> right, down -> left, right -> down)

'''


def build_sign(action_planning):
    '''
    random put a direction sign at the observation area when change direction
    :param action_planning:
    :return:
    '''
    space = [0]
    action_wanted = [action_planning[0]]
    result = np.zeros(len(action_planning), dtype=np.int)
    for i in range(len(action_planning) - 1):
        if action_planning[i] != action_planning[i + 1]:
            space.append(i + 1)
            action_wanted.append(action_planning[i + 1])
    for i in range(len(space) - 1):
        index = np.random.randint(low=space[i], high=space[i + 1], size=1)
        # 5 left turn
        # 6 right turn
        # action: 0 -> up, 1 -> down, 2 -> left, 3 -> right
        # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
        action_cur = action_wanted[i]
        action_next = action_wanted[i + 1]
        if ((action_cur == 2 and action_next == 1) or (action_cur == 1 and action_next == 3) or (
                action_cur == 3 and action_next == 0) or (action_cur == 0 and action_next == 2)):
            # make left turn
            result[index] = 5
        else:
            # make right turn
            result[index] = 6
    return result

def sign2grid(grid, start, goal, distance_sign=5):
    path, action_planning = compute_action_planning(grid, start, goal)
    sign_list = build_sign(action_planning)
    timestep = 0
    while (timestep < len(action_planning)):
        action = action_planning[timestep]
        position = path[timestep]
        sign = sign_list[timestep]
        _partial_grid = partial_grid(grid, position, distance_sign)
        if sign != 0:
            # 2 up, 3 down, 4 left, 5 right
            pos_loc = np.argwhere(_partial_grid == 1)
            np.random.shuffle(pos_loc)
            grid[pos_loc[0]] = sign
    return grid
