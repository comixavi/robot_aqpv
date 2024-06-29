import numpy as np
import matplotlib.pyplot as plt
import random

from util_ import bresenham_line


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
        new_point = (min(max(new_point[0], 0), self.grid.shape[0] - 1),
                     min(max(new_point[1], 0), self.grid.shape[1] - 1))
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
                    path.append(self.start)
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


if __name__ == '__main__':
    grid = np.array([
        [1, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1]
    ])
    start = (0, 0)
    goal = (4, 4)

    rrt = RRT(grid, start, goal)
    path_ret = rrt.find_path()
    rrt.visualize_tree(path_ret)
