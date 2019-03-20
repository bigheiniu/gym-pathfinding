import random
import numpy as np
import operator

from gym_pathfinding.games.astar import astar

def partial_grid(grid, center, observable_depth):
    """return the centered partial state, place -1 to non-visible cells"""

    i, j = center
    offset = observable_depth

    mask = np.ones_like(grid, dtype=bool)
    mask[max(0, i - offset): i + offset + 1, max(0, j - offset): j + offset + 1] = False

    _grid = np.array(grid, copy=True)
    _grid[mask] = -1
    return _grid

def generate_grid(shape, grid_type="free", generation_seed=None, spawn_seed=None):
    """ 
    Generate a grid

    shape : (lines, columns)
    grid_type : {"free", "obstacle", "maze") 

    return : grid, start, goal
    """

    if grid_type == "obstacle":
        while True:
            grid = create_obstacle(shape, generation_seed=generation_seed)
            start, goal = spawn_start_goal(grid, spawn_seed=spawn_seed)

            if path_exists(grid, start, goal):
                grid = sign2grid(grid, start, goal)
                return grid, start, goal

    grid = {
        "free" : init_grid(shape),
        "maze" : create_maze(shape, generation_seed=generation_seed),
    }[grid_type]

    start, goal = spawn_start_goal(grid, spawn_seed=spawn_seed)
    # ATTENTION: add sign to the wall
    grid = sign2grid(grid, start, goal)
    return grid, start, goal

def spawn_start_goal(grid, spawn_seed=None):
    """Returns two random position on the grid."""

    xs, ys = np.where(grid == 0)
    free_positions = list(zip(xs, ys))

    start, goal = random.Random(spawn_seed).sample(free_positions, 2)

    return start, goal

def init_grid(shape):
    grid = np.zeros(shape, dtype=np.int8)

    # Add borders
    grid[0, :] = grid[-1, :] = 1
    grid[:, 0] = grid[:, -1] = 1
    return grid

def create_maze(shape, generation_seed=None, complexity=.75, density=.50):
    # Only odd shapes
    shape = ((shape[0] // 2) * 2 + 1, (shape[1] // 2) * 2 + 1)

    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1])))
    density = int(density * ((shape[0] // 2) * (shape[1] // 2)))

    rng = np.random.RandomState(generation_seed)

    grid = init_grid(shape)

    # Make aisles
    for i in range(density):
        x, y = rng.random_integers(0, shape[1] // 2) * 2, rng.random_integers(0, shape[0] // 2) * 2
        grid[y, x] = 1

        for j in range(complexity):
            neighbours = []
            if x > 1:
                neighbours.append((y, x - 2))
            if x < shape[1] - 2:
                neighbours.append((y, x + 2))
            if y > 1:
                neighbours.append((y - 2, x))
            if y < shape[0] - 2:
                neighbours.append((y + 2, x))
            if len(neighbours):
                y_, x_ = neighbours[rng.random_integers(0, len(neighbours) - 1)]

                if grid[y_, x_] == 0:
                    grid[y_, x_] = 1
                    grid[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return grid


def create_obstacle(shape, generation_seed=None):
    rng = random.Random(generation_seed)
    lines, columns = shape
    nb_rectangles = rng.randint(3, 6)

    grid = init_grid(shape)
    for _ in range(nb_rectangles):
        add_rectangle(grid, rect(rng, lines, columns))

    return grid

def rect(rng, lines, columns):
    """ return i, j, width, height"""

    w = rng.randint(1, max(1, lines // 2))
    h = rng.randint(1, max(1, columns // 2))

    i = rng.randint(0, lines - h)
    j = rng.randint(0, columns - w)
    
    return i, j, w, h


def add_rectangle(grid, rectangle):
    i, j, w, h = rectangle

    mask = np.zeros_like(grid, dtype=bool)
    mask[i: i+h, j: j+w] = True
    grid[mask] = 1


# North, South, East, West
MOUVEMENT = [(-1, 0), (1, 0), (0, -1), (0, 1)]

def is_legal(grid, next_x, next_y):
    return grid[next_x, next_y] == 0


def path_exists(grid, start, goal):
    """ 
    Test if a path exist from start to goal
    It's a Depth-first Search
    """

    stack = [(start, [start])]

    visited = set()
    while stack:
        (vertex, path) = stack.pop()
        visited.add(vertex)

        legal_cells = set(legal_directions(grid, *vertex)) - visited
        for next in legal_cells:
            if next == goal:
                return True
            stack.append((next, path + [next]))

    return False

def legal_directions(grid, posx, posy):
    possible_moves = [(posx + dx, posy + dy) for dx, dy in MOUVEMENT]
    return [(next_x, next_y) for next_x, next_y in possible_moves if is_legal(grid, next_x, next_y)]


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


def build_sign(action_planning, path, distance):
    '''
    random put a direction sign at the observation area when change direction
    :param action_planning:
    :return:
    '''
    space = [0]
    action_wanted = [action_planning[0]]
    # result = np.zeros(len(action_planning), dtype=np.int)
    result = []
    for i in range(len(action_planning) - 1):
        if action_planning[i] != action_planning[i + 1]:
            space.append(i + 1)
            action_wanted.append(action_planning[i + 1])
    sign_path = []
    for i in range(len(space) - 1):
        value = min(space[i+1] - space[i], distance) + space[i]
        index = np.random.randint(low=space[i], high=value, size=1)[0]
        # 5 left turn
        # 6 right turn
        # action: 0 -> up, 1 -> down, 2 -> left, 3 -> right
        # mouvement = (-1, 0), (1, 0), (0, -1), (0, 1)
        action_cur = action_wanted[i]
        action_next = action_wanted[i + 1]
        if ((action_cur == 2 and action_next == 1) or (action_cur == 1 and action_next == 3) or (
                action_cur == 3 and action_next == 0) or (action_cur == 0 and action_next == 2)):
            # make left turn
            result.append(5)
            sign_path.append(path[index])
        else:
            # make right turn
            result.append(6)
            sign_path.append(path[index])


    return result, sign_path

def sign2grid(grid, start, goal, distance_sign=1):
    path, action_planning = compute_action_planning(grid, start, goal)
    sign_list, path = build_sign(action_planning, path, distance_sign)
    grid[start[0], start[1]] = -2
    grid[goal[1], goal[1]] = 4
    for index in range(len(sign_list)):
        location = path[index]
        x = location[0]
        y = location[1]
        if x == start[0] and y == start[1]:
            grid[x, y] = 7
        else:
            direction = sign_list[index]
            grid[x,y] = direction

    return grid

