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
    if isinstance(node1, NodeRRTStar) or isinstance(node1, NodeRRT) or isinstance(node2, NodeRRTStar) or isinstance(node2, NodeRRT):
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


def rrt(grid, start, goal, lim=1_000):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    max_dist = 10

    current_pos = NodeRRT(*start)
    moves = [current_pos]

    while counter < lim:
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
                    points_in = bresenham_line(path[ind][1], path[ind][0], path[ind+1][1], path[ind+1][0])

                    for point_in in points_in[1:]:
                        final_pt.append((point_in[1], point_in[0]))
                        x = point_in[1]
                        y = point_in[0]
                        if grid[x, y] == MapState.OBSTACLE.value or grid[x, y] == MapState.EXTENDED_OBSTACLE.value:
                            retry = True
                if not retry:
                    return counter, final_pt

    return None


def rrt_connect(grid, start, goal, lim=1_000):
    tr1 = [NodeRRTStar(*start)]
    tr2 = [NodeRRTStar(*goal)]
    trees = [tr1, tr2]
    # max_dist = 10
    counter = 0
    while counter < lim:
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
                    return counter*2, pt
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

    # print("")
    # print(path1)
    # print(path2)
    # print(path)

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


def chain_rrt_star(new_node, nearest_nd):
    new_node.parent = nearest_nd
    if nearest_nd is not None:
        nearest_nd.children.append(new_node)
        new_node.cost = nearest_nd.cost + distance(new_node, nearest_nd)

    return new_node


def rewire_rrt_star(tree, new_node, max_radius):
    for node in tree:
        if distance(node, new_node) < max_radius:
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                new_node.parent = node
                new_node.cost = new_cost


def rewire_rrt_star_with_check(grid, tree, new_node, max_radius=36, check_fct=obstacle_collide_bresenham_line):
    flag = False
    for node in tree:
        if distance(node, new_node) < max_radius and not check_fct(grid, new_node.x, new_node.y, node.x, node.y):  # obstacle_collide_with__checks
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                flag = True
                node.children.append(new_node)
                new_node.cost = new_cost
                new_node.parent = node
    return flag


def rrt_star(grid, start, goal, lim=1_000, check_fct=obstacle_collide_bresenham_line):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    current_pos_x = start[1]
    current_pos_y = start[0]

    current_pos = NodeRRTStar(start[0], start[1])
    moves = [current_pos]
    max_radius = 36
    max_dist = 6

    while counter < lim:
        new_pose = random_rrt_pos(current_pos_x, current_pos_y, rows, cols, max_dist)

        new_pose_x, new_pose_y = new_pose

        if (grid[new_pose_y, new_pose_x] == MapState.OBSTACLE.value or
                grid[new_pose_y, new_pose_x] == MapState.EXTENDED_OBSTACLE.value):
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose_x, new_pose_y)

        if dist > max_dist:
            new_pose.x, new_pose.y = points[max_dist - 1]
            points = points[0:max_dist]

        for point in points:
            point_y, point_x = point
            if (grid[point_y, point_x] == MapState.OBSTACLE.value or
                    grid[point_y, point_x] == MapState.EXTENDED_OBSTACLE.value):
                continue

        # if check_fct(grid, new_pose, nearest_nd):
        #    continue

        chain_rrt_star(new_pose, nearest_nd)

        # rewire_rrt_star(moves, current_pos, 2)
        rewire_rrt_star_with_check(grid, moves, new_pose, max_radius, check_fct)
        moves.append(new_pose)
        current_pos = new_pose

        counter += 1

        if goal[0] == new_pose.x and goal[1] == new_pose.y:
            path = []
            current_node = new_pose

            while current_node.parent is not None:
                points = bresenham_line(current_node.x, current_node.y, current_node.parent.x, current_node.parent.y)

                for point in points:
                    path.append((point[1], point[0]))

                current_node = current_node.parent
            path.append((start[0], start[1]))
            path.reverse()

            return counter, path

    return None


def rrt_div(grid, start, goal, lim=1_000):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    current_pos = NodeRRTStar(start[0], start[1])
    checked_combos = dict()
    moves = [current_pos]

    while counter < lim:
        min_cost = float('inf')
        new_pose = None

        points = random_quad(current_pos.x, current_pos.y, 0, 0, cols-1, rows-1)

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

        chain_rrt_star(new_pose, nearest_nd)

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
                return counter, final_pt
            else:
                current_pos = NodeRRTStar(start[0], start[1])
                moves = [current_pos]
    return None


def rrt_fast(grid, start, goal, lim=10_000):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    current_pos = NodeRRTStar(start[0], start[1])
    checked_combos = dict()
    node_goal = NodeRRTStar(goal[0], goal[1])
    moves = [current_pos]

    obs_fct = obstacle_collide_bresenham_line

    while counter < lim:
        nb_of_new_poses = 0
        nb_of_target_poses = 3
        min_cost = float('inf')
        new_pose = None

        while nb_of_new_poses < nb_of_target_poses:
            local_pose = random_rrt_pos(NodeRRTStar(current_pos), rows, cols, 9)

            if local_pose is None:
                continue

            if obs_fct(grid, local_pose, current_pos, checked_combos):
                continue

            nb_of_new_poses += 1
            local_cost = distance(local_pose, node_goal)
            if local_cost < min_cost:
                min_cost = local_cost
                new_pose = NodeRRTStar(local_pose.x, local_pose.y)

        if new_pose is None:
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose)

        if dist == 0:
            continue

        if obs_fct(grid, new_pose, nearest_nd, checked_combos):
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

            return counter, path

    return None


def plot_grid(grid, path, text):
    colors = ['blue', 'white', 'black', (0.6, 0.6, 0.6), 'green']
    Ts = 1/24
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
        nb_of_nodes, path_astar = path

        path_astar = np.array(path_astar)
        plt.plot(path_astar[:, 1], path_astar[:, 0], color='lime', marker='o')
        legend_labels.append(f"{text} Path")
        # noinspection PyTypeChecker
        legend_handles.append(
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label=f'{text} Path'))
        print(f"{text} Solution found with {nb_of_nodes} searches")
    else:
        print(f"{text} Solution isn't found")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.draw()
    plt.grid()
    plt.tight_layout()
    plt.pause(Ts)
    plt.clf()


def main():
    shape_lines = 5
    shape_col = 5
    grid = np.full((shape_lines, shape_col), MapState.FREE.value, dtype=int)

    robot_pos = (0, 0)
    obstacles = [(2, 3), (2, 2)]
    goal = (4, 4)

    grid[robot_pos] = MapState.ROBOT.value
    grid[goal] = MapState.GOAL.value

    for obstacle in obstacles:
        grid[obstacle] = MapState.OBSTACLE.value

    nb_of_test = 1000

    plot_grid(grid=grid, path=astar(grid, robot_pos, goal), text='A*')

    for i in range(nb_of_test):
        # plot_grid(grid=grid, path=rrt(grid, robot_pos, goal), text='RRT') #GOOD#
        # plot_grid(grid=grid, path=rrt_div(grid, robot_pos, goal), text='RRT Divided*')  #GOOD#
        # plot_grid(grid=grid, path=rrt_connect(grid, robot_pos, goal), text='RRT Connect')  #GOOD#

        plot_grid(grid=grid, path=rrt_star(grid, robot_pos, goal), text='RRT*')  # OBJECT COLLIDE
        # plot_grid(grid=grid, path=rrt_fast(grid, robot_pos, goal), text='RRT fast')
        print(i)

    pass


main()
