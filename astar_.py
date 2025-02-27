import heapq
import time

from mapstate_ import MapState
import numpy as np


def distance(node1, node2):
    return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def total_distance(points):
    total = 0
    for i in range(len(points) - 1):
        total += distance(points[i], points[i + 1])
    return total


def heuristic_cost_estimate(current, goal):
    return np.sqrt((current[0] - goal[0]) ** 2 + (current[1] - goal[1]) ** 2)


def reconstruct_path(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(current)
    return path[::-1]


def astar(grid, start, goal, lim=10_000, max_t=10_000_000):
    rows, cols = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic_cost_estimate(start, goal)}
    counter = 0
    start_time = time.time_ns()
    last_move = None

    while open_set and (counter < lim and (time.time_ns() - start_time) < max_t):
        if not open_set:
            return None

        _, current = heapq.heappop(open_set)

        possible_moves = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, 1), (-1, -1), (1, -1)]
        counter += 1

        if current == goal:
            path = reconstruct_path(came_from, current)
            return counter, path, (time.time_ns() - start_time) / 1e9, total_distance(path)

        for dx, dy in possible_moves:
            neighbor = (current[0] + dx, current[1] + dy)

            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols:
                if grid[neighbor[0]][neighbor[1]] in (MapState.OBSTACLE.value, MapState.EXTENDED_OBSTACLE.value):
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score

                    off_f = 1 if (dx, dy) == last_move else 1.25
                    last_move = (dx, dy)

                    f_score[neighbor] = tentative_g_score + off_f * heuristic_cost_estimate(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    return None
