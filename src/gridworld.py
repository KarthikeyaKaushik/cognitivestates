from typing import Tuple, List, Dict
import itertools
import heapq
import matplotlib.pyplot as plt

ACTIONS = ['up', 'down', 'left', 'right']

def neighbors(pos, k, l):
    i, j = pos
    moves = []
    if i > 0: moves.append(('up', (i-1, j)))
    if i < k-1: moves.append(('down', (i+1, j)))
    if j > 0: moves.append(('left', (i, j-1)))
    if j < l-1: moves.append(('right', (i, j+1)))
    return moves

def shortest_path(start, end, k, l):
    # BFS for shortest path, deterministic order
    queue = [(0, [], start)]
    visited = set()
    while queue:
        dist, path, pos = heapq.heappop(queue)
        if pos == end:
            return path
        if pos in visited:
            continue
        visited.add(pos)
        for action, npos in sorted(neighbors(pos, k, l)):
            if npos not in visited:
                heapq.heappush(queue, (dist+1, path+[action], npos))
    return None

def generate_gridworld_paths(k: int, l: int, key: Tuple[int,int], door: Tuple[int,int], goal: Tuple[int,int]) -> Dict[Tuple[int,int], List[str]]:
    """
    For each possible start location (not key, door, goal), generate the optimal path:
    - Go to key, then door, then goal, with minimal steps.
    - Path is a list of actions.
    """
    all_positions = set(itertools.product(range(k), range(l)))
    forbidden = {key, door, goal}
    paths = {}
    for start in sorted(all_positions - forbidden):
        path1 = shortest_path(start, key, k, l)
        path2 = shortest_path(key, door, k, l)
        path3 = shortest_path(door, goal, k, l)
        if path1 is None or path2 is None or path3 is None:
            raise ValueError(f'No valid path from {start} via key/door to goal')
        full_path = path1 + path2 + path3
        paths[start] = full_path
    return paths

def generate_gridworld_io_pairs(k: int, l: int, key: Tuple[int,int], door: Tuple[int,int], goal: Tuple[int,int]) -> List[Tuple[Tuple[str, ...], Tuple[str, ...]]]:
    """
    For each possible start location, generate the token-based input-output pair.
    - Input: (start_location, "n", "n", ...)
    - Output: ("n", action1, action2, ...) where actions are "u", "d", "l", "r".
    Only optimal paths are included.
    """
    paths = generate_gridworld_paths(k, l, key, door, goal)
    pairs = []
    action_map = {'up': 'u', 'down': 'd', 'left': 'l', 'right': 'r'}

    for start, actions in paths.items():
        # First input is the start location, subsequent inputs are "n"
        start_token = f"({start[0]},{start[1]})"
        input_tokens = (start_token,) + tuple("n" for _ in actions)

        # First output is "n", subsequent outputs are the actions
        output_actions = tuple(action_map[act] for act in actions)
        output_tokens = ("n",) + output_actions
        
        pairs.append((input_tokens, output_tokens))
        
    return pairs

def plot_gridworld(k: int, l: int, key: Tuple[int,int], door: Tuple[int,int], goal: Tuple[int,int], start: Tuple[int,int], path: List[str]):
    """
    Visualize the grid, special locations, start, and the path taken by the agent.
    """
    grid = [['' for _ in range(l)] for _ in range(k)]
    grid[key[0]][key[1]] = 'K'
    grid[door[0]][door[1]] = 'D'
    grid[goal[0]][goal[1]] = 'G'
    grid[start[0]][start[1]] = 'S'

    # Track agent's trajectory
    pos = list(start)
    trajectory = [tuple(pos)]
    for action in path:
        if action == 'up': pos[0] -= 1
        elif action == 'down': pos[0] += 1
        elif action == 'left': pos[1] -= 1
        elif action == 'right': pos[1] += 1
        trajectory.append(tuple(pos))

    fig, ax = plt.subplots(figsize=(l, k))
    ax.set_xlim(-0.5, l-0.5)
    ax.set_ylim(-0.5, k-0.5)
    ax.set_xticks(range(l))
    ax.set_yticks(range(k))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)

    # Draw special cells
    for i in range(k):
        for j in range(l):
            cell = grid[i][j]
            if cell:
                ax.text(j, k-1-i, cell, ha='center', va='center', fontsize=16, color='red' if cell in 'KDG' else 'blue', fontweight='bold')

    # Draw trajectory
    traj_x = [p[1] for p in trajectory]
    traj_y = [k-1-p[0] for p in trajectory]
    ax.plot(traj_x, traj_y, marker='o', color='green', linewidth=2, markersize=8)
    ax.text(traj_x[0], traj_y[0], 'S', ha='center', va='center', fontsize=12, color='blue')
    ax.text(traj_x[-1], traj_y[-1], 'E', ha='center', va='center', fontsize=12, color='green')

    plt.show()