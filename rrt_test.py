import numpy as np
import matplotlib.pyplot as plt
import random


def bresenham_line(x0, y0, x1, y1):
    dx = x1 - x0
    dy = y1 - y0
    xsign = 1 if dx > 0 else -1
    ysign = 1 if dy > 0 else -1

    # Truncate dx and dy if they exceed the maximum integer value
    max_int = 2 ** 31 - 1  # Maximum value for a 32-bit signed integer
    if abs(dx) > max_int:
        dx = max_int * xsign
    if abs(dy) > max_int:
        dy = max_int * ysign

    dx = int(abs(dx))
    dy = int(abs(dy))

    if dx > dy:
        xx, xy, yx, yy = xsign, 0, 0, ysign
    else:
        dx, dy = dy, dx
        xx, xy, yx, yy = 0, ysign, xsign, 0
    D = 2 * dy - dx
    y = 0
    for x in range(dx + 1):
        yield x0 + x * xx + y * yx, y0 + x * xy + y * yy
        if D >= 0:
            y += 1
            D -= 2 * dx
        D += 2 * dy


class RRT:
    def __init__(self, grid, start, goal, step_size=1, max_iters=1000):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.step_size = step_size
        self.max_iters = max_iters

        self.tree = {start: None}

    def generate_random_point(self):
        x = random.randint(0, self.grid.shape[0] - 1)
        y = random.randint(0, self.grid.shape[1] - 1)
        return x, y

    def nearest_neighbor(self, point):
        distances = [(np.linalg.norm(np.array(point) - np.array(node)), node) for node in self.tree.keys()]
        return min(distances, key=lambda x: x[0])[1]

    def new_point(self, nearest, random_point):
        direction = np.array(random_point) - np.array(nearest)
        distance = np.linalg.norm(direction)

        if distance == 0:
            return nearest

        direction = (direction / distance * min(distance, self.step_size)).astype(int)

        new_point = tuple(np.array(nearest) + direction)
        return new_point

    def is_obstacle_free(self, point1, point2):
        x0, y0 = point1
        x1, y1 = point2
        for x, y in bresenham_line(x0, y0, x1, y1):
            if self.grid[x, y] == 0:
                return False
        return True

    def find_path(self):
        for _ in range(self.max_iters):
            random_point = self.generate_random_point()
            nearest = self.nearest_neighbor(random_point)
            new_point = self.new_point(nearest, random_point)

            if self.is_obstacle_free(nearest, new_point):
                self.tree[new_point] = nearest

                if np.linalg.norm(np.array(new_point) - np.array(self.goal)) < self.step_size:
                    path = [self.goal]
                    while path[-1] != self.start:
                        path.append(self.tree[path[-1]])
                    return path[::-1]

        return None

    def visualize_tree(self, path=None):
        plt.imshow(self.grid, cmap='binary_r', origin='upper', interpolation='nearest')
        for point, parent in self.tree.items():
            if parent:
                plt.plot([point[1], parent[1]], [point[0], parent[0]], color='cyan', alpha=0.5)
        if path:
            path = np.array(path)
            plt.plot(path[:, 1], path[:, 0], color='lime', marker='o')
        plt.axis('off')
        plt.show()


grid = np.array([
    [1, 1, 1, 1, 1],
    [1, 0, 1, 0, 1],
    [1, 1, 1, 1, 1],
    [0, 0, 0, 1, 0],
    [1, 1, 1, 1, 1]
])
start = (0, 0)
goal = (4, 4)

rrt = RRT(grid, start, goal)
path = rrt.find_path()
rrt.visualize_tree(path)
