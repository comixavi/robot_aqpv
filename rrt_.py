import random
import time
from multiprocessing import freeze_support

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
from datetime import datetime
from mapstate_ import MapState
from util_ import bresenham_line
from astar_ import astar
from genetic_ import genetic_algorithm


# from xlsxwriter import Workbook


class NodeRRT:
    def __init__(self, x_or_node, y=None):
        if y is None:
            self.x = x_or_node.x
            self.y = x_or_node.y
        else:
            self.x = x_or_node
            self.y = y
        self.parent = None

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


class NodeRRTStar:
    def __init__(self, x_or_node, y=None):
        if y is None:
            self.x = x_or_node.x
            self.y = x_or_node.y
            self.parent = None
            self.cost = 0
            self.children = []
        else:
            self.x = x_or_node
            self.y = y
            self.parent = None
            self.cost = 0
            self.children = []

    def __str__(self):
        return f"x: {self.x}, y: {self.y}"

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def random_pos(rows, cols):
    x = random.randint(0, cols - 1)
    y = random.randint(0, rows - 1)
    # print(rows, "x", cols, " ", y, "x", x)

    return x, y


def random_rrt_pos(current_pos_x, current_pos_y, rows, cols, max_dist=2):
    x = random.randint(max(current_pos_x - max_dist, 0), min(current_pos_x + max_dist, cols - 1))
    y = random.randint(max(current_pos_y - max_dist, 0), min(current_pos_y + max_dist, rows - 1))

    return x, y


def random_quad(x, y, min_x, min_y, max_x, max_y):
    p1_x = random.randint(x, max_x)
    p1_y = random.randint(y, max_y)

    p2_x = random.randint(min_x, x)
    p2_y = random.randint(y, max_y)

    p3_x = random.randint(min_x, x)
    p3_y = random.randint(min_y, y)

    p4_x = random.randint(x, max_x)
    p4_y = random.randint(min_y, y)

    return (p1_x, p1_y), (p2_x, p2_y), (p3_x, p3_y), (p4_x, p4_y)


def obstacle_collide(grid, new_pos):
    pose_y = new_pos[0]
    pose_x = new_pos[1]

    return grid[pose_x, pose_y] == MapState.OBSTACLE.value or grid[pose_x, pose_y] == MapState.EXTENDED_OBSTACLE.value


def obstacle_collide_with__checks(grid, current_pos, new_pose, checked_combos=None):
    if current_pos is None or new_pose is None:
        return False

    if checked_combos is not None:
        if (current_pos.x, current_pos.y, new_pose.x, new_pose.y,) in checked_combos.keys():
            return True

    if new_pose.x == current_pos.x and new_pose.y == current_pos.y:
        return True

    if new_pose.x == current_pos.x:  # x1 = x2 -> same j (cols);
        start_i = min(new_pose.y, current_pos.y)
        end_i = max(new_pose.y, current_pos.y)
        j = new_pose.x
        for i in range(start_i, end_i + 1):
            if grid[i, j] == MapState.OBSTACLE.value or grid[i, j] == MapState.EXTENDED_OBSTACLE.value:
                if checked_combos is not None:
                    checked_combos[(current_pos.x, current_pos.y, new_pose.x, new_pose.y)] = True
                return True

    if new_pose.y == current_pos.y:
        start_j = min(new_pose.x, current_pos.x)
        end_j = max(new_pose.x, current_pos.x)
        i = new_pose.y
        for j in range(start_j, end_j + 1):
            if grid[i, j] == MapState.OBSTACLE.value or grid[i, j] == MapState.EXTENDED_OBSTACLE.value:
                if checked_combos is not None:
                    checked_combos[(current_pos.x, current_pos.y, new_pose.x, new_pose.y)] = True
                return True

    if new_pose.y != current_pos.y and new_pose.x != current_pos.x:
        start_i = min(new_pose.y, current_pos.y)
        end_i = max(new_pose.y, current_pos.y)

        start_j = min(new_pose.x, current_pos.x)
        end_j = max(new_pose.x, current_pos.x)

        for i in range(start_i, end_i + 1):
            for j in range(start_j, end_j + 1):
                if grid[i, j] == MapState.OBSTACLE.value or grid[i, j] == MapState.EXTENDED_OBSTACLE.value:
                    if checked_combos is not None:
                        checked_combos[(current_pos.x, current_pos.y, new_pose.x, new_pose.y)] = True
                    return True

    return False


def obstacle_collide_bresenham_line(grid, current_pos_x, current_pos_y, new_pose_x, new_pose_y, checked_combos=None):
    if checked_combos is not None:
        if (current_pos_x, current_pos_y, new_pose_x, new_pose_y,) in checked_combos.keys():
            return True

    if new_pose_x == current_pos_x and new_pose_y == current_pos_y:
        return True

    points = bresenham_line(current_pos_x, current_pos_y, new_pose_x, new_pose_y)

    for point in points:
        i = point[0]
        j = point[1]

        # j = min(point[0], len(grid)-1)
        # i = min(point[1], len(grid[0])-1)

        if point[0] > len(grid) - 1 or point[1] > len(grid[0]) - 1:
            return True

        if grid[i, j] == MapState.OBSTACLE.value or grid[i, j] == MapState.EXTENDED_OBSTACLE.value:
            if checked_combos is not None:
                checked_combos[(current_pos_x, current_pos_y, new_pose_x, new_pose_y)] = True
            return True


def obstacle_collide_l_check(grid, current_pos, new_pose, checked_combos=None):
    if current_pos is None or new_pose is None:
        return False

    if checked_combos is not None:
        if (current_pos.x, current_pos.y, new_pose.x, new_pose.y,) in checked_combos.keys():
            return True

    if new_pose.x == current_pos.x and new_pose.y == current_pos.y:
        return True

    start_i = min(new_pose.y, current_pos.y)
    end_i = max(new_pose.y, current_pos.y)

    start_j = min(new_pose.x, current_pos.x)
    end_j = max(new_pose.x, current_pos.x)

    while start_i != end_i:
        end_i -= 1

        i = end_i
        j = start_j

        if grid[i, j] == MapState.OBSTACLE.value or \
                grid[i, j] == MapState.EXTENDED_OBSTACLE.value:
            if checked_combos is not None:
                checked_combos[(current_pos.x, current_pos.y, new_pose.x, new_pose.y)] = True
            return True

    while start_j != end_j:
        start_j += 1

        i = end_i
        j = start_j

        if grid[i, j] == MapState.OBSTACLE.value or \
                grid[i, j] == MapState.EXTENDED_OBSTACLE.value:
            if checked_combos is not None:
                checked_combos[(current_pos.x, current_pos.y, new_pose.x, new_pose.y)] = True
            return True

    return False


def distance(node1, node2):
    if isinstance(node1, NodeRRTStar) or isinstance(node1, NodeRRT) or isinstance(node2, NodeRRTStar) or isinstance(
            node2, NodeRRT):
        return np.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)
    else:
        return np.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)


def total_distance(points):
    total = 0
    for i in range(len(points) - 1):
        total += distance(points[i], points[i + 1])
    return total


def nearest_node(vertices, new_node_x, new_node_y):
    min_distance = float('inf')
    nearest_nd = None
    min_path = None

    for vertex in vertices:
        loc_path = bresenham_line(vertex.x, vertex.y, new_node_x, new_node_y)
        dist = len(loc_path)

        if dist < min_distance and dist != 0:
            min_distance = dist
            nearest_nd = vertex
            min_path = loc_path

    return min_distance, nearest_nd, min_path


# def nearest_nodes(vertices, new_node, max_distance=2):
#     nearest_nd = []
#     for vertex in vertices:
#         dist = distance(vertex, new_node)
#         if dist < max_distance:
#             nearest_nd.append(vertex)
#     return nearest_nd


def rrt(grid, start, goal, lim=1_000, max_t=10):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    max_dist = 32

    goal = goal[1], goal[0]

    current_pos = NodeRRT(start[1], start[0])
    moves = [current_pos]
    start_time = time.time_ns()
    # goal/start = y, x

    while counter < lim and time.time_ns() - start_time < max_t:
        counter += 1

        new_pose = random_pos(rows, cols)
        new_pose_x, new_pose_y = new_pose

        if obstacle_collide(grid, new_pose):
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose_x, new_pose_y)

        if dist == 0:
            continue

        if dist > max_dist:
            # new_pose_x, new_pose_y = points[max_dist]
            points = points[0:max_dist]

        break_key = False
        for point in points:
            y = point[0]
            x = point[1]

            if grid[y, x] == MapState.OBSTACLE.value or grid[y, x] == MapState.EXTENDED_OBSTACLE.value:
                break_key = True
                break

        if break_key:
            continue

        for point in points:
            new_node = NodeRRTStar(point[1], point[0])
            new_node.parent = nearest_nd
            moves.append(new_node)

            if goal[1] == point[0] and goal[0] == point[1]:
                path = [goal]
                current_node = new_node

                while current_node is not None:
                    path.append((current_node.y, current_node.x))
                    current_node = current_node.parent

                # while current_node.parent is not None:
                #     points_in = bresenham_line(current_node.x, current_node.y, current_node.parent.x, current_node.parent.y)
                #
                #     for point_in in points_in[1:]:
                #         path.append((point_in[1], point_in[0]))
                #
                #     current_node = current_node.parent

                path.reverse()
                final_pt = [start]
                retry = False
                for ind, local_pt in enumerate(path[:-2]):
                    points_in = bresenham_line(path[ind][1], path[ind][0], path[ind + 1][1], path[ind + 1][0])

                    for point_in in points_in[1:]:
                        final_pt.append((point_in[0], point_in[1]))
                        x = point_in[1]
                        y = point_in[0]
                        if grid[y, x] == MapState.OBSTACLE.value or grid[y, x] == MapState.EXTENDED_OBSTACLE.value:
                            retry = True
                if not retry:
                    return counter, final_pt, time.time_ns() - start_time, total_distance(final_pt)

    return None


def reconstruct_path_con(grid, node_from_tr1, node_from_tr2):
    path1 = []
    current = node_from_tr1
    while current:
        path1.append((current.x, current.y))
        current = current.parent

    path2 = []
    current = node_from_tr2

    while current:
        path2.append((current.x, current.y))
        current = current.parent

    path = (path1[::-1] + path2[1:])[::-1]

    final_pt = [path[0]]

    for ind, local_pt in enumerate(path[:-1]):
        points_in = bresenham_line(path[ind][0], path[ind][1], path[ind + 1][0], path[ind + 1][1])

        for point_in in points_in:
            final_pt.append((point_in[0], point_in[1]))
            x = point_in[0]
            y = point_in[1]
            if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                return None
    final_pt[0] = final_pt[0][::-1]

    final_pt.append(path[-1][::-1])

    return final_pt


def rrt_connect(grid, start, goal, lim=1_000, max_t=10):
    tr1 = [NodeRRTStar(*start[::-1])]
    tr2 = [NodeRRTStar(*goal[::-1])]

    trees = [tr1, tr2]

    counter = 0
    start_time = time.time_ns()
    # print(len(grid), len(grid[0]))
    # print("Start: ", start)
    # print("Goal: ", goal)

    while counter < lim and time.time_ns() - start_time < max_t:
        counter += 1

        for tree in trees:
            random_node = random_pos(len(grid), len(grid[0]))
            #print("Punctul random: ", random_node)

            if obstacle_collide(grid, random_node):
                continue

            dist, nearest_nd, path = nearest_node(tree, *random_node)
            # print("Cel mai aproape nod: ", nearest_nd)
            # print("")
            if dist == 0:
                continue

            # if dist > max_dist:
            #     path = path[:max_dist + 1]
            #     random_node = (path[-1][0], path[-1][1])

            break_key = False
            for point in path:
                y = point[0]
                x = point[1]

                # print(point, end=" ")
                if grid[y, x] == MapState.OBSTACLE.value or grid[y, x] == MapState.EXTENDED_OBSTACLE.value:
                    break_key = True
                    break
            # print("\n")

            if break_key:
                continue

            new_node = NodeRRTStar(*random_node)
            new_node.parent = nearest_nd
            tree.append(new_node)

            other_tree = tr2 if tree is tr1 else tr1
            dist, other_nearest, path = nearest_node(other_tree, new_node.x, new_node.y)

            break_key = False
            for point in path:
                y = point[0]
                x = point[1]
                # print(point, end=" ")

                if grid[y, x] in (MapState.OBSTACLE.value, MapState.EXTENDED_OBSTACLE.value):
                    break_key = True
                    break
            # print("\n")

            if break_key:
                continue

            if new_node == other_nearest:
                if tr2 == tree:
                    pt = reconstruct_path_con(grid, new_node, other_nearest)
                else:
                    pt = reconstruct_path_con(grid, other_nearest, new_node)

                if pt is not None:
                    return counter * 2, pt, time.time_ns() - start_time, total_distance(pt)
                else:
                    tr1 = [NodeRRTStar(*start[::-1])]
                    tr2 = [NodeRRTStar(*goal[::-1])]
                    trees = [tr1, tr2]
    return None


def rewire_rrt_star_with_check(grid, tree, new_node, max_radius=36, check_fct=obstacle_collide_bresenham_line):
    flag = False
    for node in tree:
        if distance(node, new_node) < max_radius and not check_fct(grid, new_node.x, new_node.y, node.x,
                                                                   node.y):  # obstacle_collide_with__checks
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                flag = True
                node.children.append(new_node)
                new_node.cost = new_cost
                new_node.parent = node
    return flag


def chain_rrt_star(new_node, nr_node):
    if nr_node is not None:
        new_node.parent = nr_node
        new_node.cost = nr_node.cost + distance(new_node, nr_node)
        nr_node.children.append(new_node)


def rewire_rrt_star(tree, new_node, max_radius):
    for node in tree:
        if distance(node, new_node) < max_radius:
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                if node.parent:
                    node.parent.children.remove(node)
                node.parent = new_node
                new_node.children.append(node)
                node.cost = new_cost
                update_children_costs(node)


def update_children_costs(node):
    for child in node.children:
        new_cost = node.cost + distance(node, child)
        if new_cost < child.cost:
            child.cost = new_cost
            update_children_costs(child)


def random_rrt_pos_star(rows, cols):
    return (
        random.randint(0, cols - 1),
        random.randint(0, rows - 1)
    )


def random_rrt_pos_max_dist(x, y, rows, cols, max_dist):
    if max(x - max_dist, 0) > min(x + max_dist, cols - 1):
        return None
    if max(y - max_dist, 0) > min(y + max_dist, rows - 1):
        return None

    return (
        random.randint(max(x - max_dist, 0), min(x + max_dist, cols - 1)),
        random.randint(max(y - max_dist, 0), min(y + max_dist, rows - 1))
    )


def obstacle_collide_star(grid, x, y):
    return grid[y][x] in (MapState.OBSTACLE.value, MapState.EXTENDED_OBSTACLE.value)


def nearest_node_star(tree, x, y):
    nearest = None
    min_distance = float('inf')

    for node in tree:
        dist = np.sqrt((node.x - x) ** 2 + (node.y - y) ** 2)
        if dist < min_distance:
            nearest = node
            min_distance = dist

    return nearest, min_distance


def construct_path_star(grid, node):
    path = []

    while node:
        path.append((node.x, node.y))
        node = node.parent

    path = path[::-1]

    final_pt = []

    for ind, local_pt in enumerate(path[:-1]):
        points_in = bresenham_line(path[ind][0], path[ind][1], path[ind + 1][0], path[ind + 1][1])

        for point_in in points_in:
            final_pt.append((point_in[0], point_in[1]))

            y = point_in[0]
            x = point_in[1]

            if grid[y, x] in (MapState.OBSTACLE.value, MapState.EXTENDED_OBSTACLE.value):
                return None

    final_pt.append(path[-1])

    return final_pt


def rrt_star(grid, start, goal, lim=1000, max_dist=10, max_t=10):
    rows, cols = grid.shape
    tree = [NodeRRTStar(*start[::-1])]
    counter = 0
    start_time = time.time_ns()

    while counter < lim and time.time_ns() - start_time < max_t:
        counter += 1

        rand_x, rand_y = random_rrt_pos_star(rows, cols)

        if obstacle_collide_star(grid, rand_x, rand_y):
            continue

        nr_node, _ = nearest_node_star(tree, rand_x, rand_y)

        if nr_node is None:
            continue

        new_node = NodeRRTStar(rand_x, rand_y)

        chain_rrt_star(new_node, nr_node)
        rewire_rrt_star(tree, new_node, max_dist)
        tree.append(new_node)

        if (new_node.y, new_node.x) == goal:
            rec_path = construct_path_star(grid, new_node)
            if rec_path is not None:
                return counter, rec_path, time.time_ns() - start_time, total_distance(rec_path)
            else:
                tree = [NodeRRTStar(*start[::-1])]

    return None


def chain_rrt_div(new_node, nearest_nd):
    new_node.parent = nearest_nd
    if nearest_nd is not None:
        nearest_nd.children.append(new_node)
        new_node.cost = nearest_nd.cost + distance(new_node, nearest_nd)

    return new_node


def rrt_div(grid, start, goal, lim=1_000, max_t=10):
    rows, cols = grid.shape
    counter = 0
    current_pos = NodeRRTStar(*start[::-1])
    checked_combos = dict()
    moves = [current_pos]
    goal = goal[::-1]

    start_time = time.time_ns()
    dst = 5
    while counter < lim and time.time_ns() - start_time < max_t:
        min_cost = float('inf')
        new_pose = None

        points = random_quad(current_pos.x, current_pos.y, max(0, current_pos.x - dst), max(0, current_pos.y - dst),
                             min(current_pos.x + dst, cols - 1), min(current_pos.y + dst, cols - 1))

        for point in points:
            if obstacle_collide_bresenham_line(grid, *point, current_pos.x, current_pos.y, checked_combos):
                continue

            local_cost = distance(point, goal)
            if local_cost < min_cost:
                min_cost = local_cost
                new_pose = NodeRRTStar(point[0], point[1])

        if new_pose is None:
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose.x, new_pose.y)

        if dist == 0:
            continue

        for point in points:
            y = point[0]
            x = point[1]
            if grid[y, x] == MapState.OBSTACLE.value or grid[y, x] == MapState.EXTENDED_OBSTACLE.value:
                continue

        chain_rrt_div(new_pose, nearest_nd)

        current_pos = new_pose

        moves.append(new_pose)

        counter += 1

        if goal[0] == new_pose.x and goal[1] == new_pose.y:
            path = []
            current_node = new_pose

            while current_node is not None:
                path.append((current_node.y, current_node.x))
                current_node = current_node.parent

            # path.append((start[0], start[1]))
            path.reverse()
            final_pt = [start]
            retry = False

            for ind, local_pt in enumerate(path[:-1]):
                points_in = bresenham_line(path[ind][0], path[ind][1], path[ind + 1][0], path[ind + 1][1])

                for point_in in points_in:
                    final_pt.append((point_in[1], point_in[0]))
                    y = point_in[0]
                    x = point_in[1]
                    if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                        retry = True

            if not retry:
                return counter, final_pt, time.time_ns() - start_time, total_distance(final_pt)
            else:
                current_pos = NodeRRTStar(start[1], start[0])
                moves = [current_pos]
    return None


def random_pos_fast(counter_close, goal, rows, cols):
    p = 0.10 if counter_close < 3 else 0.25 if counter_close < 5 else 0.50
    # print(p)
    if random.randint(1, 100) / 100 < p:
        return random_rrt_pos_max_dist(goal[1], goal[0], rows, cols, int(1 / p))
    else:
        return random_pos(rows, cols)
    pass


def rrt_fast(grid, start, goal, lim=1_000, max_t=10):
    rows, cols = grid.shape
    counter = 0
    counter_close = 0
    current_pos = NodeRRTStar(*start[::-1])
    checked_combos = dict()
    moves = [current_pos]

    start_time = time.time_ns()

    while counter < lim and time.time_ns() - start_time < max_t:
        counter += 1

        local_pose = random_pos_fast(counter_close, goal, rows, cols)
        counter_close += 1

        if local_pose is None:
            continue

        # if obstacle_collide_bresenham_line(grid, *local_pose, current_pos.x, current_pos.y, checked_combos):
        #     counter_close = 0
        #     continue

        # nb_of_new_poses += 1
        # local_cost = distance(local_pose, goal)
        #
        # if local_cost < min_cost:
        #     min_cost = local_cost
        #     new_pose = NodeRRTStar(*local_pose)
        #
        # if new_pose is None:
        #     continue

        new_pose = NodeRRTStar(*local_pose)
        dist, nearest_nd, points = nearest_node(moves, *local_pose)

        if dist == 0:
            continue

        if obstacle_collide_bresenham_line(grid, *local_pose, nearest_nd.x, nearest_nd.y, checked_combos):
            counter_close = 0
            continue

        chain_rrt_star(new_pose, nearest_nd)

        # current_pos = new_pose

        moves.append(new_pose)

        counter += 1

        if goal[1] == new_pose.x and goal[0] == new_pose.y:
            path = []
            current_node = new_pose

            # while current_node is not None:
            #     path.append((current_node.y, current_node.x))
            #     current_node = current_node.parent

            while current_node.parent is not None:
                points = bresenham_line(current_node.x, current_node.y, current_node.parent.x, current_node.parent.y)

                for point in points:
                    path.append((point[1], point[0]))

                current_node = current_node.parent

            # cost = current_node.cost
            path.append((start[1], start[0]))
            path.reverse()

            final_pt = [start]
            retry = False

            for ind, local_pt in enumerate(path[:-2]):
                points_in = bresenham_line(path[ind][1], path[ind][0], path[ind + 1][1], path[ind + 1][0])

                for point_in in points_in[1:]:
                    final_pt.append((point_in[1], point_in[0]))
                    x = point_in[0]
                    y = point_in[1]
                    if grid[y, x] == MapState.OBSTACLE.value or grid[y, x] == MapState.EXTENDED_OBSTACLE.value:
                        retry = True

            final_pt.append(goal)

            if not retry:
                return counter, final_pt, time.time_ns() - start_time, total_distance(final_pt)
            else:
                current_pos = NodeRRTStar(start[0], start[1])
                counter_close = 0
                checked_combos = dict()
                moves = [current_pos]

    return None


def generate_grid_with_obstacles(num_obstacles):
    grid = np.full((20, 20), MapState.FREE.value)

    def can_place_obstacle():
        if y + length > 20:
            return False
        for i in range(length):
            if grid[x, y + i] != MapState.FREE.value:
                return False
            if x > 0 and grid[x - 1, y + i] == MapState.OBSTACLE.value:
                return False
            if x < 19 and grid[x + 1, y + i] == MapState.OBSTACLE.value:
                return False
        return True

    for _ in range(num_obstacles):
        placed = False
        while not placed:
            x = random.randint(0, 19)
            y = random.randint(0, 19)
            length = random.randint(2, 10)

            if can_place_obstacle():
                for i in range(length):
                    grid[x, y + i] = MapState.OBSTACLE.value
                placed = True

    return grid


def plot_blank_grid(grid, title):
    colors = ['blue', 'white', 'black', (0.6, 0.6, 0.6), 'green']
    Ts = 100
    map_cmap = ListedColormap(colors, name='color_map')
    bounds = [i + 1 for i in range(len(colors) + 1)]
    map_norm = BoundaryNorm(bounds, len(bounds) - 1)

    y_max, x_max = len(grid), len(grid[0])
    x_min = -1
    y_min = -1

    plt.imshow(grid, cmap=map_cmap, norm=map_norm, interpolation='none')
    legend_labels = [state.name for state in MapState]
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                    color=map_cmap(map_norm(i + 1))) for i in range(len(legend_labels))]

    plt.title(f'Robot Environment - {title}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.draw()
    plt.grid()
    plt.tight_layout()
    plt.pause(Ts)
    plt.clf()


def plot_grid(grid, path, text):
    colors = ['blue', 'white', 'black', (0.6, 0.6, 0.6), 'green']
    Ts = 1/144
    map_cmap = ListedColormap(colors, name='color_map')
    bounds = [i + 1 for i in range(len(colors) + 1)]
    map_norm = BoundaryNorm(bounds, len(bounds) - 1)

    y_max, x_max = len(grid), len(grid[0])
    x_min = -1
    y_min = -1

    plt.imshow(grid, cmap=map_cmap, norm=map_norm, interpolation='none')
    legend_labels = [state.name for state in MapState]
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                    color=map_cmap(map_norm(i + 1))) for i in range(len(legend_labels))]

    plt.title(f'Robot Environment - {text}')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    color = (28 / 255, 43 / 255, 244 / 255, 1)

    if path is not None:
        nb_of_nodes, path, times, costs = path

        path = np.array(path)
        plt.plot(path[:, 1], path[:, 0], color=color, marker='o')
        legend_labels.append(f"{text} Path")
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color,
                       markersize=10, label=f'{text} Path'))

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.draw()
    plt.grid()
    plt.tight_layout()
    plt.pause(Ts)
    plt.clf()


def print_statistics(data, title):
    data_array = np.array(data)

    print(f"Method: {title}")

    nan_count = np.isnan(data_array).sum()
    print(f"Number of fails: {nan_count}")

    clean_data = data_array[~np.isnan(data_array)]

    if clean_data.size > 0:
        mean = np.mean(clean_data)

        std_dev = np.std(clean_data)

        median = np.median(clean_data)

        min_val = np.min(clean_data)
        max_val = np.max(clean_data)

        # Print the calculated statistics
        print(f"Mean: {mean:.2f}")
        print(f"Standard Deviation: {std_dev:.2f}")
        print(f"Median: {median:.2f}")
        print(f"Minimum: {min_val:.2f}")
        print(f"Maximum: {max_val:.2f}")
    else:
        print("No valid data available for statistics after removing NaN values.")
    print("")


def read_from_excel(filename):
    df = pd.read_excel(filename, header=None, sheet_name='Results')

    results = {}
    current_method = None
    category = None

    for index, row in df.iterrows():
        if pd.isna(row[0]):
            continue

        if pd.isna(row[1]):
            current_method = row[0]
            results[current_method] = {}
        elif current_method:
            category = row[0]
            values = row[1:].dropna().tolist()
            values = [None if v == 'NaN' else v for v in values]
            results[current_method][category] = values

    return results


def calculate_number_of_turns(path):
    turns = 0
    if len(path) < 3:
        return turns

    for i in range(1, len(path) - 1):
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]
        dx2 = path[i + 1][0] - path[i][0]
        dy2 = path[i + 1][1] - path[i][1]

        if dx1 != dx2 or dy1 != dy2:
            turns += 1

    return turns


def calculate_smoothness(path):
    angles = []
    if len(path) < 3:
        return 0

    for i in range(1, len(path) - 1):
        dx1 = path[i][0] - path[i - 1][0]
        dy1 = path[i][1] - path[i - 1][1]
        dx2 = path[i + 1][0] - path[i][0]
        dy2 = path[i + 1][1] - path[i + 1][1]

        dot_product = dx1 * dx2 + dy1 * dy2
        magnitude1 = np.sqrt(dx1 ** 2 + dy1 ** 2)
        magnitude2 = np.sqrt(dx2 ** 2 + dy2 ** 2)

        if magnitude1 * magnitude2 == 0:
            continue

        angle = np.arccos(dot_product / (magnitude1 * magnitude2))
        angles.append(angle)

    return np.mean(angles) if angles else 0


def plot_comparison(methods, path_lengths, number_of_turns, smoothness, execution_times, energy_costs, fail_nb,
                    criteria):
    data = [
        path_lengths,
        number_of_turns,
        smoothness,
        execution_times,
        energy_costs,
        fail_nb
    ]

    num_methods = len(methods)
    num_criteria = len(criteria)

    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.15
    index = np.arange(num_methods)

    for i in range(num_criteria):
        ax.bar(index + i * bar_width, data[i], bar_width, label=criteria[i])

    ax.set_xlabel('Methods')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Methods by Various Criteria')
    ax.set_xticks(index + bar_width * (num_criteria - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.xaxis.set_tick_params(labelsize=12)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_comparison_individual(methods, path_lengths, number_of_turns, smoothness, execution_times, energy_costs,
                               fail_nb, criteria):
    data = [
        path_lengths,
        number_of_turns,
        smoothness,
        execution_times,
        energy_costs,
        fail_nb
    ]

    num_methods = len(methods)
    num_criteria = len(criteria)

    # Iterate over each method and create a separate figure for each
    for j in range(num_methods):
        fig, ax = plt.subplots(figsize=(8, 6))  # Smaller figure size for individual plots

        bar_width = 0.5
        index = np.arange(num_criteria)  # Index for criteria

        # Plot bars for each criterion for the current method
        for i in range(num_criteria):
            ax.bar(index[i] + bar_width, data[i][j], bar_width, label=criteria[i])

        ax.set_xlabel('Criteria')
        ax.set_ylabel('Values')
        ax.set_title(f'Comparison of {methods[j]} by Various Criteria')
        ax.set_xticks(index + bar_width)
        ax.set_xticklabels(criteria)  # Label with criterion names
        ax.legend()

        plt.tight_layout()
        plt.show()


def eliminate_duplicates(lst):
    if not lst:
        return []

    result = [lst[0]]
    last_element = lst[0]

    for i in range(1, len(lst)):
        if lst[i] != last_element:
            result.append(lst[i])
        last_element = lst[i]

    return result


def plot_deviations(methods, data, criteria, mean_values):
    num_methods = len(methods)
    num_criteria = len(criteria)
    fig, ax = plt.subplots(figsize=(12, 8))

    bar_width = 0.15
    index = np.arange(num_methods)

    for i, criterion in enumerate(criteria):
        if mean_values[i] != 0:  # Avoid division by zero
            deviations = [abs(data[j][i] - mean_values[i]) / mean_values[i] for j in range(num_methods)]
        else:
            deviations = [0] * num_methods  # Set deviations to 0 where mean is 0 to handle division by zero
        for ind, dev in enumerate(deviations):
            if dev > 1:
                deviations[ind] = 1
        ax.bar(index + i * bar_width, deviations, bar_width, label=f'Standardized Deviation of {criterion}')

    ax.set_xlabel('Methods')
    ax.set_ylabel('Standardized Deviation')
    ax.set_title('Standardized Deviations from Mean by Various Criteria')
    ax.set_xticks(index + bar_width * (num_criteria - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    def s2ns(sec):
        return sec*1_000_000_000

    simple_grid = False
    random_grid = False
    lidar_grid = True
    grid = None

    robot_pos = None
    goal = None

    frame_by_frame = True
    use_ga = False
    use_astar = True
    use_astar_stats = False

    if simple_grid:
        shape_lines = 5
        shape_col = 5
        grid = np.full((shape_lines, shape_col), MapState.FREE.value, dtype=int)
        robot_pos = (0, 0)
        obstacles = [(3, 3), (2, 2)]
        goal = (4, 4)

        grid[robot_pos] = MapState.ROBOT.value
        grid[goal] = MapState.GOAL.value

        for obstacle in obstacles:
            grid[obstacle] = MapState.OBSTACLE.value

    if lidar_grid:
        robot_pos = (5, 10)
        goal = (20, 20)

        grid_aux = [
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 4, 4, 4, 4, 4, 2, 4, 3, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 3, 3, 4, 4, 3, 3, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 3, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 4, 3, 4, 4, 4, 3, 4, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 4, 4, 3, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 3, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 4, 3, 3, 4, 3, 4, 2, 2, 2, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 3, 3, 3, 4, 4, 2, 2, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 3, 3, 3, 4, 4, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 4, 3, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 4, 4, 3, 3, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 4, 4, 4, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 2, 4, 3, 4, 4, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 3, 4, 4, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 4, 3, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 4, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 3, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 3, 3, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 3, 3, 3, 3, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 3, 4, 3, 3, 3, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 4, 4, 4, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 4, 3, 4, 2, 2, 2, 2, 4, 4, 2, 2, 2],
            [2, 2, 4, 3, 4, 3, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 3, 4, 3, 3, 3, 3, 4, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 4, 3, 3, 4, 2, 2, 2, 4, 3, 4, 2, 2, 2],
            [2, 2, 4, 3, 4, 3, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4, 4, 3, 3, 3, 4, 4, 4, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 4, 4, 4, 2, 2, 4, 4, 4, 3, 4, 2, 2, 2, 2],
            [2, 2, 4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 2, 2, 2, 2, 4, 4, 4, 2, 4, 3, 3, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 4, 4, 2, 2, 2, 4, 3, 4, 4, 4, 2, 2, 2, 2, 2],
            [2, 2, 2, 4, 4, 3, 4, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 3, 3, 4, 4, 4, 3, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 4, 3, 4, 4, 4, 2, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 4, 3, 4, 3, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 3, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 4, 4, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 4, 4, 4, 4, 2, 2, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 4, 3, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
             2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
        ]

        shape_lines = len(grid_aux)
        shape_col = len(grid_aux[0])

        grid = np.full((shape_lines, shape_col), MapState.FREE.value, dtype=int)

        for i in range(shape_lines - 1):
            for j in range(shape_col - 1):
                grid[i, j] = grid_aux[i][j]

        grid[robot_pos] = MapState.ROBOT.value
        grid[goal] = MapState.GOAL.value

    if random_grid:
        grid = generate_grid_with_obstacles(5)
        robot_pos = (0, 0)
        goal = (19, 19)

        grid[robot_pos] = MapState.ROBOT.value
        grid[goal] = MapState.GOAL.value

    nb_of_test = 10

    rrt_nb_list, rrt_nb_of_turns, rrt_smoothness, rrt_nb_t_exec, rrt_nb_cost, rrt_nb_fails = [], [], [], [], [], 0
    rrt_fast_list, rrt_fast_nb_of_turns, rrt_fast_smoothness, rrt_fast_t_exec, rrt_fast_cost, rrt_fast_fails = [], [], [], [], [], 0
    rrt_div_list, rrt_div_nb_of_turns, rrt_div_smoothness, rrt_div_t_exec, rrt_div_cost, rrt_div_fails = [], [], [], [], [], 0
    rrt_con_list, rrt_con_nb_of_turns, rrt_con_smoothness, rrt_con_t_exec, rrt_con_cost, rrt_con_fails = [], [], [], [], [], 0
    rrt_star_list, rrt_star_nb_of_turns, rrt_star_smoothness, rrt_star_t_exec, rrt_star_cost, rrt_star_fails = [], [], [], [], [], 0

    ga_list, ga_nb_of_turns, ga_smoothness, ga_t_exec, ga_cost, ga_fails = [], [], [], [], [], 0
    astar_nb_list, astar_nb_of_turns, astar_smoothness, astar_nb_t_exec, astar_nb_cost, astar_nb_fails = [], [], [], [], [], 0

    max_time = s2ns(4)
    lim_iter = 500_000

    if use_astar:
        astar_sol = astar(grid, robot_pos, goal)

        if astar_sol is not None:
            if frame_by_frame:
                for j in range(len(astar_sol[1]) - 1):
                    rrt_sol_aux = astar_sol[0], astar_sol[1][0:j + 1], astar_sol[2], astar_sol[3]
                    plot_grid(grid=grid, path=rrt_sol_aux, text='A*')
            else:
                plot_grid(grid=grid, path=astar_sol, text='A*')
                astar_nb_list.append(len(astar_sol[1]))
            astar_nb_t_exec.append(1)  # astar_sol[2] / 1_000_000)
            astar_nb_cost.append(astar_sol[3])
            astar_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(astar_sol[1])))
            astar_smoothness.append(calculate_smoothness(eliminate_duplicates(astar_sol[1])))
        else:
            astar_nb_list.append(np.nan)
            astar_nb_t_exec.append(np.nan)
            astar_nb_cost.append(np.nan)
            astar_nb_of_turns.append(np.nan)
            astar_smoothness.append(np.nan)
            astar_nb_fails += 1

    for i in range(nb_of_test):
        print(i / nb_of_test)

        if random_grid:
            grid = generate_grid_with_obstacles(5)
            robot_pos = (0, 0)
            goal = (19, 19)

            grid[robot_pos] = MapState.ROBOT.value
            grid[goal] = MapState.GOAL.value

        if use_astar and random_grid:
            astar_sol = astar(grid, robot_pos, goal)

            if astar_sol is not None:
                if frame_by_frame:
                    for j in range(len(astar_sol[1]) - 1):
                        rrt_sol_aux = astar_sol[0], astar_sol[1][0:j + 1], astar_sol[2], astar_sol[3]
                        plot_grid(grid=grid, path=rrt_sol_aux, text='A*')
                else:
                    plot_grid(grid=grid, path=astar_sol, text='A*')
                    astar_nb_list.append(len(astar_sol[1]))
                astar_nb_t_exec.append(1)  # astar_sol[2] / 1_000_000)
                astar_nb_cost.append(astar_sol[3])
                astar_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(astar_sol[1])))
                astar_smoothness.append(calculate_smoothness(eliminate_duplicates(astar_sol[1])))
            else:
                astar_nb_list.append(np.nan)
                astar_nb_t_exec.append(np.nan)
                astar_nb_cost.append(np.nan)
                astar_nb_of_turns.append(np.nan)
                astar_smoothness.append(np.nan)
                astar_nb_fails += 1

        rrt_sol = rrt(grid, robot_pos, goal, lim_iter, max_time)
        if rrt_sol is not None:
            if frame_by_frame:
                for j in range(len(rrt_sol[1]) - 1):
                    rrt_sol_aux = rrt_sol[0], rrt_sol[1][0:j + 1], rrt_sol[2], rrt_sol[3]
                    plot_grid(grid=grid, path=rrt_sol_aux, text='RRT')
            else:
                plot_grid(grid=grid, path=rrt_sol, text='RRT')

            rrt_nb_list.append(len(rrt_sol[1]))
            rrt_nb_t_exec.append(rrt_sol[2] / 1_000_000)
            rrt_nb_cost.append(rrt_sol[3])
            rrt_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(rrt_sol[1])))
            rrt_smoothness.append(calculate_smoothness(eliminate_duplicates(rrt_sol[1])))
        else:
            rrt_nb_list.append(np.nan)
            rrt_nb_t_exec.append(np.nan)
            rrt_nb_cost.append(np.nan)
            rrt_nb_of_turns.append(np.nan)
            rrt_smoothness.append(np.nan)
            rrt_nb_fails += 1

        rrt_con_sol = rrt_connect(grid, robot_pos, goal, lim_iter, max_time)
        if rrt_con_sol is not None:
            if frame_by_frame:
                for j in range(len(rrt_con_sol[1]) - 1):
                    rrt_con_sol_aux = rrt_con_sol[0], rrt_con_sol[1][0:j + 1], rrt_con_sol[2], rrt_con_sol[3]
                    plot_grid(grid=grid, path=rrt_con_sol_aux, text='RRT Connect')
            else:
                plot_grid(grid=grid, path=rrt_con_sol, text='RRT Connect')

            rrt_con_list.append(len(rrt_con_sol[1]))
            rrt_con_t_exec.append(rrt_con_sol[2] / 1_000_000)
            rrt_con_cost.append(rrt_con_sol[3])
            rrt_con_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(rrt_con_sol[1])))
            rrt_con_smoothness.append(calculate_smoothness(eliminate_duplicates(rrt_con_sol[1])))
        else:
            rrt_con_list.append(np.nan)
            rrt_con_t_exec.append(np.nan)
            rrt_con_cost.append(np.nan)
            rrt_con_nb_of_turns.append(np.nan)
            rrt_con_smoothness.append(np.nan)
            rrt_con_fails += 1

        rrt_div_sol = rrt_div(grid, robot_pos, goal, lim_iter, max_time)
        if rrt_div_sol is not None:
            if frame_by_frame:
                for j in range(len(rrt_div_sol[1]) - 1):
                    rrt_div_sol_aux = rrt_div_sol[0], rrt_div_sol[1][0:j + 1], rrt_div_sol[2], rrt_div_sol[3]
                    plot_grid(grid=grid, path=rrt_div_sol_aux, text='RRT div')
            else:
                plot_grid(grid=grid, path=rrt_div_sol, text='RRT div')
            rrt_div_list.append(len(rrt_div_sol[1]))
            rrt_div_t_exec.append(rrt_div_sol[2] / 1_000_000)
            rrt_div_cost.append(rrt_div_sol[3])
            rrt_div_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(rrt_div_sol[1])))
            rrt_div_smoothness.append(calculate_smoothness(eliminate_duplicates(rrt_div_sol[1])))
        else:
            rrt_div_list.append(np.nan)
            rrt_div_t_exec.append(np.nan)
            rrt_div_cost.append(np.nan)
            rrt_div_nb_of_turns.append(np.nan)
            rrt_div_smoothness.append(np.nan)
            rrt_div_fails += 1

        rrt_fast_sol = rrt_fast(grid, robot_pos, goal, lim_iter, max_time)
        if rrt_fast_sol is not None:
            if frame_by_frame:
                for j in range(len(rrt_fast_sol[1]) - 1):
                    rrt_fast_sol_aux = rrt_fast_sol[0], rrt_fast_sol[1][0:j + 1], rrt_fast_sol[2], rrt_fast_sol[3]
                    plot_grid(grid=grid, path=rrt_fast_sol_aux, text='RRT fast')
            else:
                plot_grid(grid=grid, path=rrt_fast_sol, text='RRT fast')

            rrt_fast_list.append(len(rrt_fast_sol[1]))
            rrt_fast_t_exec.append(rrt_fast_sol[2] / 1_000_000)
            rrt_fast_cost.append(rrt_fast_sol[3])
            rrt_fast_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(rrt_fast_sol[1])))
            rrt_fast_smoothness.append(calculate_smoothness(eliminate_duplicates(rrt_fast_sol[1])))
        else:
            rrt_fast_list.append(np.nan)
            rrt_fast_t_exec.append(np.nan)
            rrt_fast_cost.append(np.nan)
            rrt_fast_nb_of_turns.append(np.nan)
            rrt_fast_smoothness.append(np.nan)
            rrt_fast_fails += 1

        max_dist = 10
        rrt_star_sol = rrt_star(grid, robot_pos, goal, lim_iter, max_dist, max_time)
        if rrt_star_sol is not None:
            if frame_by_frame:
                for j in range(len(rrt_star_sol[1]) - 1):
                    rrt_star_sol_aux = rrt_star_sol[0], rrt_star_sol[1][0:j + 1], rrt_star_sol[2], rrt_star_sol[3]
                    plot_grid(grid=grid, path=rrt_star_sol_aux, text='RRT*')
            else:
                plot_grid(grid=grid, path=rrt_star_sol, text='RRT*')

            rrt_star_list.append(len(rrt_star_sol[1]))
            rrt_star_t_exec.append(rrt_star_sol[2] / 1_000_000)
            rrt_star_cost.append(rrt_star_sol[3])
            rrt_star_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(rrt_star_sol[1])))
            rrt_star_smoothness.append(calculate_smoothness(eliminate_duplicates(rrt_star_sol[1])))
        else:
            rrt_star_list.append(np.nan)
            rrt_star_t_exec.append(np.nan)
            rrt_star_cost.append(np.nan)
            rrt_star_nb_of_turns.append(np.nan)
            rrt_star_smoothness.append(np.nan)
            rrt_star_fails += 1

        if use_ga and simple_grid:
            lens = [i * 4 + 5 for i in range(int(len(grid) * len(grid[0]) / 4))]
            ga_solution = None
            for ln in lens:
                ga_solution = genetic_algorithm(population_size=1000, move_length=ln, generations=100,
                                                mutation_rate=0.25, grid=grid, max_time=max_time)
                if ga_solution is not None:
                    break

            if ga_solution is not None:
                loc_point = robot_pos
                path_ga = [loc_point]
                for ind, move in enumerate(ga_solution[1]):
                    loc_goal = (path_ga[ind][0] + move[0], path_ga[ind][1] + move[1])
                    path_ga.append(loc_goal)
                ga_solution = ga_solution[0], path_ga, ga_solution[2], ga_solution[3]

                if frame_by_frame:
                    for j in range(len(path_ga) - 1):
                        rrt_ga_aux = ga_solution[0], path_ga[0:j + 1], ga_solution[2], ga_solution[3]
                        plot_grid(grid=grid, path=rrt_ga_aux, text='GA')
                else:
                    plot_grid(grid=grid, path=ga_solution, text='GA')
                    ga_list.append(len(path_ga[1]))
                ga_t_exec.append(ga_solution[2] / 1_000_000)
                ga_cost.append(ga_solution[3])
                ga_nb_of_turns.append(calculate_number_of_turns(eliminate_duplicates(path_ga)))
                ga_smoothness.append(calculate_smoothness(eliminate_duplicates(path_ga)))
            else:
                ga_list.append(np.nan)
                ga_t_exec.append(np.nan)
                ga_cost.append(np.nan)
                ga_nb_of_turns.append(np.nan)
                ga_smoothness.append(np.nan)
                ga_fails += 1

    methods = ['RRT', 'RRT Connect', 'RRT Div', 'RRT Fast', 'RRT Star']

    path_lengths = [
        np.nanmean(rrt_nb_list), np.nanmean(rrt_con_list), np.nanmean(rrt_div_list),
        np.nanmean(rrt_fast_list), np.nanmean(rrt_star_list)
    ]
    number_of_turns = [
        np.nanmean(rrt_nb_of_turns), np.nanmean(rrt_con_nb_of_turns),
        np.nanmean(rrt_div_nb_of_turns), np.nanmean(rrt_fast_nb_of_turns), np.nanmean(rrt_star_nb_of_turns)
    ]
    smoothness = [
        np.nanmean(rrt_smoothness), np.nanmean(rrt_con_smoothness),
        np.nanmean(rrt_div_smoothness), np.nanmean(rrt_fast_smoothness), np.nanmean(rrt_star_smoothness)
    ]
    execution_times = [
        np.nanmean(rrt_nb_t_exec), np.nanmean(rrt_con_t_exec), np.nanmean(rrt_div_t_exec),
        np.nanmean(rrt_fast_t_exec), np.nanmean(rrt_star_t_exec)
    ]
    energy_costs = [
        np.nanmean(rrt_nb_cost), np.nanmean(rrt_con_cost), np.nanmean(rrt_div_cost),
        np.nanmean(rrt_fast_cost), np.nanmean(rrt_star_cost)
    ]
    number_of_fails = [
        rrt_nb_fails / nb_of_test * 100, rrt_con_fails / nb_of_test * 100, rrt_div_fails / nb_of_test * 100,
        rrt_fast_fails / nb_of_test * 100, rrt_star_fails / nb_of_test * 100
    ]

    if use_astar_stats:
        methods.append('A Star')
        path_lengths.append(np.nanmean(astar_nb_list))
        number_of_turns.append(np.nanmean(astar_nb_of_turns))
        smoothness.append(np.nanmean(astar_smoothness))
        execution_times.append(np.nanmean(astar_nb_t_exec))
        energy_costs.append(np.nanmean(astar_nb_cost))
        number_of_fails.append(astar_nb_fails / nb_of_test * 100)

    if use_ga:
        methods.append('GA')
        path_lengths.append(np.nanmean(ga_list))
        number_of_turns.append(np.nanmean(ga_nb_of_turns))
        smoothness.append(np.nanmean(ga_smoothness))
        execution_times.append(np.nanmean(ga_t_exec))
        energy_costs.append(np.nanmean(ga_cost))
        number_of_fails.append(ga_fails / nb_of_test * 100)

    criteria = ['Number of Nodes', 'Number of Turns', 'Smoothness',
                'Execution Time [ms]', 'Path Length [m]', 'Percentage of Failed Tests']

    variables = [
        ('methods', methods),
        ('path_lengths', path_lengths),
        ('number_of_turns', number_of_turns),
        ('smoothness', smoothness),
        ('execution_times', execution_times),
        ('energy_costs', energy_costs),
        ('number_of_fails', number_of_fails),
        ('criteria', criteria)
    ]

    for name, var in variables:
        print(name, var)

    plot_comparison_individual(methods, path_lengths, number_of_turns, smoothness,
                               execution_times, energy_costs, number_of_fails, criteria)

    plot_comparison(methods, path_lengths, number_of_turns, smoothness,
                    execution_times, energy_costs, number_of_fails, criteria)

    print("NUMBER OF TRIES ANALYSIS:")
    print_statistics(rrt_nb_list, 'RRT')
    print_statistics(rrt_con_list, 'RRT Connect')
    print_statistics(rrt_div_list, 'RRT Div')
    print_statistics(rrt_fast_list, 'RRT Fast')
    if use_ga:
        print_statistics(ga_list, 'GA')
    print("\n\n\n")

    print("TIME ANALYSIS:")
    print_statistics(rrt_nb_t_exec, 'RRT')
    print_statistics(rrt_con_t_exec, 'RRT Connect')
    print_statistics(rrt_div_t_exec, 'RRT Div')
    print_statistics(rrt_fast_t_exec, 'RRT Fast')
    if use_ga:
        print_statistics(ga_t_exec, 'GA')
    print("\n\n\n")

    print("COST ANALYSIS:")
    print_statistics(rrt_nb_cost, 'RRT')
    print_statistics(rrt_con_cost, 'RRT Connect')
    print_statistics(rrt_div_cost, 'RRT Div')
    print_statistics(rrt_fast_cost, 'RRT Fast')
    if use_ga:
        print_statistics(ga_cost, 'GA')
    print("\n\n\n")

    print("NUMBER OF TURNS ANALYSIS:")
    print_statistics(rrt_nb_of_turns, 'RRT')
    print_statistics(rrt_con_nb_of_turns, 'RRT Connect')
    print_statistics(rrt_div_nb_of_turns, 'RRT Div')
    print_statistics(rrt_fast_nb_of_turns, 'RRT Fast')
    print_statistics(rrt_star_nb_of_turns, 'RRT Star')
    if use_ga:
        print_statistics(ga_nb_of_turns, 'GA')
    print("\n\n\n")

    print("SMOOTHNESS ANALYSIS:")
    print_statistics(rrt_smoothness, 'RRT')
    print_statistics(rrt_con_smoothness, 'RRT Connect')
    print_statistics(rrt_div_smoothness, 'RRT Div')
    print_statistics(rrt_fast_smoothness, 'RRT Fast')
    print_statistics(rrt_star_smoothness, 'RRT Star')
    if use_ga:
        print_statistics(ga_smoothness, 'GA')

    mean_path_lengths = np.nanmean(path_lengths)
    mean_number_of_turns = np.nanmean(number_of_turns)
    mean_smoothness = np.nanmean(smoothness)
    mean_execution_times = np.nanmean(execution_times)
    mean_energy_costs = np.nanmean(energy_costs)

    mean_values = [mean_path_lengths, mean_number_of_turns, mean_smoothness, mean_execution_times, mean_energy_costs]

    plot_deviations(methods, [path_lengths, number_of_turns, smoothness, execution_times, energy_costs], criteria[:-1],
                    mean_values)

    write_excel = False
    if write_excel:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'result{current_time}.xlsx'

        data = {
            'RRT': {
                'Path': rrt_nb_list,
                'Execution Time': rrt_nb_t_exec,
                'Cost': rrt_nb_cost
            },
            'RRT Connect': {
                'Path': rrt_con_list,
                'Execution Time': rrt_con_t_exec,
                'Cost': rrt_con_cost
            },
            'RRT Div': {
                'Path': rrt_div_list,
                'Execution Time': rrt_div_t_exec,
                'Cost': rrt_div_cost
            },
            'RRT Fast': {
                'Path': rrt_fast_list,
                'Execution Time': rrt_fast_t_exec,
                'Cost': rrt_fast_cost
            },
            'RRT star': {
                'Path': rrt_star_list,
                'Execution Time': rrt_star_t_exec,
                'Cost': rrt_star_cost
            }
        }
        if use_ga:
            data['GA'] = {
                'Path': rrt_star_list,
                'Execution Time': rrt_star_t_exec,
                'Cost': rrt_star_cost
            }

        print("Write Excel")

        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("Results")
            writer.sheets['Results'] = worksheet

            row = 0
            for method, values in data.items():
                worksheet.write(row, 0, method)
                row += 1
                for key, val_list in values.items():
                    worksheet.write(row, 0, key)
                    for col, val in enumerate(val_list, start=1):
                        if pd.isna(val):
                            worksheet.write_string(row, col, 'NaN')
                        else:
                            worksheet.write(row, col, val)
                    row += 1
                row += 1  # add a blank line after each method
        print("Ended Write Excel")


main()
