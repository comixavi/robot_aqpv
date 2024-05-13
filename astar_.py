import heapq
import time

from mapstate_ import MapState
import numpy as np


def heuristic_cost_estimate(current, goal):
    return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]


def astar(grid, start, goal, lim=10_000, max_t=10):
    rows, cols = len(grid), len(grid[0])

    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    last_move = ()
    g_score = {start: 0}
    f_score = {start: heuristic_cost_estimate(start, goal)}
    counter = 0
    start_time = time.time()

    while open_set and counter < lim or max_t > time.time() - start_time:
        _, current = heapq.heappop(open_set)
        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]

        counter += 1

        if current[0] == goal[0] and current[1] == goal[1]:
            return counter, reconstruct_path(came_from, current), time.time() - start_time

        for dx, dy in possible_moves:
            neighbor = (current[0] + dx, current[1] + dy)

            if neighbor[0] < 0 or neighbor[0] >= rows or neighbor[1] < 0 or neighbor[1] >= cols:
                continue

            if (grid[neighbor[0]][neighbor[1]] == MapState.OBSTACLE.value or
                    grid[neighbor[0]][neighbor[1]] == MapState.EXTENDED_OBSTACLE.value):
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                off_f = 1 if last_move == (dx, dy) else 1.25
                last_move = (dx, dy)
                f_score[neighbor] = tentative_g_score + off_f * heuristic_cost_estimate(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None
