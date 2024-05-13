import random
import time
from hmac import new

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm

import mapstate_
from mapstate_ import MapState
from util_ import bresenham_line
from astar_ import astar


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
        if grid[j, i] == MapState.OBSTACLE.value or grid[j, i] == MapState.EXTENDED_OBSTACLE.value:
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
    max_dist = 10

    current_pos = NodeRRT(*start)
    moves = [current_pos]
    start_time = time.time()

    while counter < lim and time.time() - start_time < max_t:
        counter += 1

        new_pose = random_pos(rows, cols)
        new_pose_x, new_pose_y = new_pose

        if obstacle_collide(grid, new_pose):
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose_x, new_pose_y)

        if dist == 0:
            continue

        if dist > max_dist:
            new_pose_x, new_pose_y = points[max_dist]
            points = points[0:max_dist + 1]

        break_key = False
        for point in points:
            y = point[0]
            x = point[1]

            if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                break_key = True
                break

        if break_key:
            continue

        for point in points:
            new_node = NodeRRTStar(point[0], point[1])
            new_node.parent = nearest_nd
            moves.append(new_node)

            if goal[0] == point[1] and goal[1] == point[0]:
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
                        final_pt.append((point_in[1], point_in[0]))
                        x = point_in[1]
                        y = point_in[0]
                        if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                            retry = True
                if not retry:
                    return counter, final_pt, time.time() - start_time

    return None


def rrt_connect(grid, start, goal, lim=1_000, max_t=10):
    tr1 = [NodeRRTStar(*start)]
    tr2 = [NodeRRTStar(*goal)]
    trees = [tr1, tr2]
    # max_dist = 10
    counter = 0
    start_time = time.time()

    while counter < lim and time.time() - start_time < max_t:
        counter += 1

        for tree in trees:
            random_node = random_pos(len(grid), len(grid[0]))

            if obstacle_collide(grid, random_node):
                continue

            dist, nearest_nd, path = nearest_node(tree, *random_node)

            if dist == 0:
                continue

            # if dist > max_dist:
            #     path = path[:max_dist + 1]
            #     random_node = (path[-1][0], path[-1][1])

            break_key = False
            for point in path:
                y = point[0]
                x = point[1]

                if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                    break_key = True
                    break

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

                if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                    break_key = True
                    break

            if break_key:
                continue

            if new_node == other_nearest:
                if tr2 == tree:
                    pt = reconstruct_path(grid, new_node, other_nearest)
                else:
                    pt = reconstruct_path(grid, other_nearest, new_node)

                if pt is not None:
                    return counter * 2, pt, time.time() - start_time
                else:
                    tr1 = [NodeRRTStar(*start)]
                    tr2 = [NodeRRTStar(*goal)]
                    trees = [tr1, tr2]
    return None


def reconstruct_path(grid, node_from_tr1, node_from_tr2):
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

    final_pt.append(path[-1])

    return final_pt


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
        nr_node.children.append(new_node)
        new_node.cost = nr_node.cost + distance(new_node, nr_node)


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
        old_cost = child.cost
        new_cost = node.cost + distance(node, child)
        if new_cost < old_cost:
            child.cost = new_cost
            update_children_costs(child)


def random_rrt_pos_star(rows, cols):
    return (
        random.randint(0, cols - 1),
        random.randint(0, rows - 1)
    )


def random_rrt_pos_max_dist(x, y, rows, cols, max_dist):
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


def rrt_star(grid, start, goal, lim=1000, max_dist=10, max_t=10):
    tree = [NodeRRTStar(*start)]
    counter = 0
    start_time = time.time()

    while counter < lim and time.time() - start_time < max_t:
        counter += 1

        rand_x, rand_y = random_rrt_pos_star(len(grid), len(grid[0]))

        if obstacle_collide_star(grid, rand_x, rand_y):
            continue

        nr_node, _ = nearest_node_star(tree, rand_x, rand_y)
        new_node = NodeRRTStar(rand_x, rand_y)
        chain_rrt_star(new_node, nr_node)
        rewire_rrt_star(tree, new_node, max_dist)
        tree.append(new_node)

        if (new_node.x, new_node.y) == goal:
            rec_path = construct_path_star(grid, new_node)
            if rec_path is not None:
                return counter, rec_path, time.time() - start_time
            else:
                tree = [NodeRRTStar(*start)]

    return None


def construct_path_star(grid, node):
    path = []

    while node:
        path.append((node.x, node.y))
        node = node.parent

    path = path[::-1]

    final_pt = [path[0]]

    for ind, local_pt in enumerate(path[:-1]):
        points_in = bresenham_line(path[ind][0], path[ind][1], path[ind + 1][0], path[ind + 1][1])

        for point_in in points_in:
            final_pt.append((point_in[0], point_in[1]))

            x = point_in[0]
            y = point_in[1]

            if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                return None

    final_pt.append(path[-1])

    return final_pt


def chain_rrt_div(new_node, nearest_nd):
    new_node.parent = nearest_nd
    if nearest_nd is not None:
        nearest_nd.children.append(new_node)
        new_node.cost = nearest_nd.cost + distance(new_node, nearest_nd)

    return new_node


def rrt_div(grid, start, goal, lim=1_000, max_t=10):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    current_pos = NodeRRTStar(start[0], start[1])
    checked_combos = dict()
    moves = [current_pos]

    start_time = time.time()

    while counter < lim and time.time() - start_time < max_t:
        min_cost = float('inf')
        new_pose = None

        points = random_quad(current_pos.x, current_pos.y, 0, 0, cols - 1, rows - 1)

        for point in points:
            if obstacle_collide_bresenham_line(grid, *point, current_pos.x, current_pos.y, checked_combos):
                continue

            local_cost = distance(point, goal)
            if local_cost < min_cost:
                min_cost = local_cost
                new_pose = NodeRRTStar(*point)

        if new_pose is None:
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose.x, new_pose.y)

        if dist == 0:
            continue

        for point in points:
            x = point[0]
            y = point[1]
            if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
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
                    final_pt.append((point_in[0], point_in[1]))
                    x = point_in[0]
                    y = point_in[1]
                    if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                        retry = True

            if not retry:
                return counter, final_pt, time.time() - start_time
            else:
                current_pos = NodeRRTStar(start[0], start[1])
                moves = [current_pos]
    return None


def random_pos_fast(counter_close, goal, rows, cols):
    p = 0.10 if counter_close < 3 else 0.25 if counter_close < 5 else 0.50

    if random.randint(1, 100) / 100 < p:
        return random_rrt_pos_max_dist(goal[0], goal[1], rows, cols, int(1 / p))
    else:
        return random_pos(rows, cols)
    pass


def rrt_fast(grid, start, goal, lim=1_000, max_t=10):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    counter_close = 0
    current_pos = NodeRRTStar(start[0], start[1])
    checked_combos = dict()
    moves = [current_pos]

    obs_fct = obstacle_collide_bresenham_line

    start_time = time.time()

    while counter < lim and time.time() - start_time < max_t:
        counter += 1

        local_pose = random_pos_fast(counter_close, goal, rows, cols)
        counter_close += 1

        if local_pose is None:
            continue

        if obs_fct(grid, *local_pose, current_pos.x, current_pos.y, checked_combos):
            counter_close = 0
            continue

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

        if obs_fct(grid, *local_pose, nearest_nd.x, nearest_nd.y, checked_combos):
            counter_close = 0
            continue

        chain_rrt_star(new_pose, nearest_nd)

        current_pos = new_pose

        moves.append(new_pose)

        counter += 1

        if goal[0] == new_pose.x and goal[1] == new_pose.y:
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

            path.append((start[0], start[1]))
            path.reverse()

            final_pt = [start]
            retry = False

            for ind, local_pt in enumerate(path[:-2]):
                points_in = bresenham_line(path[ind][1], path[ind][0], path[ind + 1][1], path[ind + 1][0])

                for point_in in points_in[1:]:
                    final_pt.append((point_in[1], point_in[0]))
                    x = point_in[1]
                    y = point_in[0]
                    if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                        retry = True

            final_pt.append(goal)

            if not retry:
                return counter, final_pt, time.time() - start_time
            else:
                current_pos = NodeRRTStar(start[0], start[1])
                counter_close = 0
                checked_combos = dict()
                moves = [current_pos]

    return None


def plot_grid(grid, path, text):
    colors = ['blue', 'white', 'black', (0.6, 0.6, 0.6), 'green']
    Ts = 1 / 24
    map_cmap = ListedColormap(colors, name='color_map')
    bounds = [i + 1 for i in range(len(colors) + 1)]
    map_norm = BoundaryNorm(bounds, len(bounds) - 1)

    y_max, x_max = grid.shape
    x_min = -1
    y_min = -1

    plt.imshow(grid, cmap=map_cmap, norm=map_norm, interpolation='none')
    legend_labels = [state.name for state in MapState]
    legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                    color=map_cmap(map_norm(i + 1))) for i in range(len(legend_labels))]

    plt.title('Occupancy Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    if path is not None:
        nb_of_nodes, path_astar, times = path

        path_astar = np.array(path_astar)
        plt.plot(path_astar[:, 1], path_astar[:, 0], color='lime', marker='o')
        legend_labels.append(f"{text} Path")
        # noinspection PyTypeChecker
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label=f'{text} Path'))
        # print(f"{text} Solution found with {nb_of_nodes} searches")
    # else:
        # print(f"{text} Solution isn't found")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.draw()
    plt.grid()
    plt.tight_layout()
    plt.pause(Ts)
    plt.clf()


def print_statistics(data, title):
    # Convert data to numpy array for efficient computation
    data_array = np.array(data)

    # Count the number of np.nan values (fails) and display them
    print(f"Method: {title}")

    nan_count = np.isnan(data_array).sum()
    print(f"Number of fails: {nan_count}")

    # Filter out np.nan values for accurate statistical calculations
    clean_data = data_array[~np.isnan(data_array)]

    # Proceed with calculations if there is clean data available
    if clean_data.size > 0:
        # Calculate mean
        mean = np.mean(clean_data)

        # Calculate standard deviation
        std_dev = np.std(clean_data)

        # Calculate median
        median = np.median(clean_data)

        # Find minimum and maximum values
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


def main():
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

    nb_of_test = 100

    plot_grid(grid=grid, path=astar(grid, robot_pos, goal), text='A*')

    rrt_nb_list = []
    rrt_nb_t_exec = []

    rrt_fast_list = []
    rrt_fast_t_exec = []

    rrt_div_list = []
    rrt_div_t_exec = []

    rrt_con_list = []
    rrt_con_t_exec = []

    rrt_star_list = []
    rrt_star_t_exec = []

    max_time = 10_000

    for i in range(nb_of_test):
        rrt_sol = rrt(grid, robot_pos, goal, max_time)
        if rrt_sol is not None:
            plot_grid(grid=grid, path=rrt_sol, text='RRT')  # GOOD#
            rrt_nb_list.append(rrt_sol[0])
            rrt_nb_t_exec.append(rrt_sol[2])
        else:
            rrt_nb_list.append(np.nan)
            rrt_nb_t_exec.append(np.nan)

        rrt_con_sol = rrt_connect(grid, robot_pos, goal, max_time)
        if rrt_con_sol is not None:
            plot_grid(grid=grid, path=rrt_con_sol, text='RRT Connect')  # GOOD#
            rrt_con_list.append(rrt_con_sol[0])
            rrt_con_t_exec.append(rrt_con_sol[2])
        else:
            rrt_con_list.append(np.nan)
            rrt_con_t_exec.append(np.nan)

        rrt_div_sol = rrt_div(grid, robot_pos, goal, max_time)
        if rrt_div_sol is not None:
            plot_grid(grid=grid, path=rrt_div_sol, text='RRT div')  # GOOD#
            rrt_div_list.append(rrt_div_sol[0])
            rrt_div_t_exec.append(rrt_div_sol[2])
        else:
            rrt_div_list.append(np.nan)
            rrt_div_t_exec.append(np.nan)

        rrt_fast_sol = rrt_fast(grid, robot_pos, goal, max_time)
        if rrt_fast_sol is not None:
            plot_grid(grid=grid, path=rrt_fast_sol, text='RRT fast')  # GOOD#
            rrt_fast_list.append(rrt_fast_sol[0])
            rrt_fast_t_exec.append(rrt_fast_sol[2])
        else:
            rrt_fast_list.append(np.nan)
            rrt_fast_t_exec.append(np.nan)

        max_dist = 100
        rrt_star_sol = rrt_star(grid, robot_pos, goal, max_dist, max_time)
        if rrt_star_sol is not None:
            plot_grid(grid=grid, path=rrt_star_sol, text='RRT*')  # GOOD#
            rrt_star_list.append(rrt_star_sol[0])
            rrt_star_t_exec.append(rrt_star_sol[2])
        else:
            rrt_star_list.append(np.nan)
            rrt_star_t_exec.append(np.nan)

        print(i/nb_of_test)

    print("NUMBER OF TRIES ANALYSIS:")
    print_statistics(rrt_nb_list, 'RRT')
    print_statistics(rrt_con_list, 'RRT Connect')
    print_statistics(rrt_div_list, 'RRT Div')
    print_statistics(rrt_fast_list, 'RRT Fast')
    print("\n\n\n")

    print("TIME ANALYSIS:")
    print_statistics(rrt_nb_t_exec, 'RRT')
    print_statistics(rrt_con_t_exec, 'RRT Connect')
    print_statistics(rrt_div_t_exec, 'RRT Div')
    print_statistics(rrt_fast_t_exec, 'RRT Fast')
    

main()
