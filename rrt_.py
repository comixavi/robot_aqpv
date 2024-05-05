import random
from mapstate_ import MapState
from util_ import bresenham_line


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


def random_rrt_pos(current_pos, rows, cols, max_dist=2, all_map=False):
    if all_map:
        x = random.randint(0, cols - 1)
        y = random.randint(0, rows - 1)
    else:
        x = random.randint(max(current_pos.x - max_dist, 0), min(current_pos.x + max_dist, cols - 1))
        y = random.randint(max(current_pos.y - max_dist, 0), min(current_pos.y + max_dist, rows - 1))

    if type(current_pos) == NodeRRT:
        return NodeRRT(x, y)
    else:
        return NodeRRTStar(x, y)


def obstacle_collide(grid, new_pos):
    return grid[new_pos.y, new_pos.x] == MapState.OBSTACLE.value or \
        grid[new_pos.y, new_pos.x] == MapState.EXTENDED_OBSTACLE.value


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


def obstacle_collide_bresenham_line(grid, current_pos, new_pose, checked_combos=None):
    if current_pos is None or new_pose is None:
        return False

    if checked_combos is not None:
        if (current_pos.x, current_pos.y, new_pose.x, new_pose.y,) in checked_combos.keys():
            return True

    if new_pose.x == current_pos.x and new_pose.y == current_pos.y:
        return True

    points = bresenham_line(current_pos.x, current_pos.y, new_pose.x, new_pose.y)

    for point in points:
        i = point[0]
        j = point[1]
        if grid[j, i] == MapState.OBSTACLE.value or grid[j, i] == MapState.EXTENDED_OBSTACLE.value:
            if checked_combos is not None:
                checked_combos[(current_pos.x, current_pos.y, new_pose.x, new_pose.y)] = True
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
    return abs(node1.x - node2.x) + abs(node1.y - node2.y)


def nearest_node(vertices, new_node):
    min_distance = float('inf')
    nearest_nd = None
    min_path = None

    for vertex in vertices:
        loc_path = bresenham_line(vertex.x, vertex.y, new_node.x, new_node.y)
        dist = len(loc_path)

        if dist < min_distance and dist != 0:
            min_distance = dist
            nearest_nd = vertex
            min_path = loc_path

    return min_distance, nearest_nd, min_path


def nearest_nodes(vertices, new_node, max_distance=2):
    nearest_nd = []
    for vertex in vertices:
        dist = distance(vertex, new_node)
        if dist < max_distance:
            nearest_nd.append(vertex)
    return nearest_nd


def rrt(grid, start, goal, lim=10_000):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    max_dist = 6

    current_pos = NodeRRT(start[0], start[1])
    moves = [current_pos]

    while counter < lim:
        new_pose = random_rrt_pos(NodeRRT(current_pos), rows, cols, max_dist, True)

        if new_pose is None:
            continue

        if obstacle_collide(grid, new_pose):
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose)

        if dist == 0:
            continue

        if dist > max_dist:
            # points = bresenham_line(nearest_nd.x, nearest_nd.y, new_pose.x, new_pose.y)
            new_pose.x, new_pose.y = points[max_dist]
            points = points[0:max_dist+1]

        break_key = False
        for point in points:
            x = point[0]
            y = point[1]
            if grid[y, x] == MapState.OBSTACLE.value or grid[y, x] == MapState.EXTENDED_OBSTACLE.value:
                break_key = True
                break

        if break_key:
            continue

        new_pose.parent = nearest_nd
        moves.append(new_pose)

        current_pos = new_pose

        counter += 1

        if goal[0] == new_pose.x and goal[1] == new_pose.y:
            path = []
            current_node = new_pose

            while current_node is not None:
                path.append((current_node.y, current_node.x))
                current_node = current_node.parent

            #path.append((start[0], start[1]))
            path.reverse()

            return counter, path

    return None


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
        if distance(node, new_node) < max_radius and not check_fct(grid, new_node, node): #obstacle_collide_with__checks
            new_cost = new_node.cost + distance(node, new_node)
            if new_cost < node.cost:
                flag = True
                node.children.append(new_node)
                new_node.cost = new_cost
                new_node.parent = node
    return flag


def rrt_star(grid, start, goal, lim=10_000, check_fct=obstacle_collide_bresenham_line):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    current_pos = NodeRRTStar(start[0], start[1])
    moves = [current_pos]
    max_radius = 36
    max_dist = 6

    while counter < lim:
        new_pose = random_rrt_pos(NodeRRTStar(current_pos), rows, cols, max_dist, True)

        if new_pose is None:
            continue

        if check_fct(grid, new_pose, current_pos):
            continue

        dist, nearest_nd, points = nearest_node(moves, new_pose)

        if dist > max_dist:
            new_pose.x, new_pose.y = points[max_dist - 1]

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

            while current_node is not None:
                path.append((current_node.y, current_node.x))
                current_node = current_node.parent
            path.append((start[0], start[1]))
            path.reverse()

            return counter, path

    return None


def my_rrt(grid, start, goal, lim=10_000):
    rows, cols = len(grid), len(grid[0])
    counter = 0
    current_pos = NodeRRTStar(start[0], start[1])
    checked_combos = dict()
    node_goal = NodeRRTStar(goal[0], goal[1])
    moves = [current_pos]

    while counter < lim:
        nb_of_new_poses = 0
        nb_of_target_poses = 3
        min_cost = float('inf')
        new_pose = None

        while nb_of_new_poses < nb_of_target_poses:
            local_pose = random_rrt_pos(NodeRRTStar(current_pos), rows, cols, 9)

            if local_pose is None:
                continue

            if obstacle_collide_with__checks(grid, local_pose, current_pos, checked_combos):
                continue

            nb_of_new_poses += 1
            local_cost = distance(local_pose, node_goal)
            if local_cost < min_cost:
                min_cost = local_cost
                new_pose = NodeRRTStar(local_pose.x, local_pose.y)

        if new_pose is None:
            continue

        dist, nearest_nd = nearest_node(moves, new_pose)

        if dist == 0:
            continue

        if obstacle_collide_with__checks(grid, new_pose, nearest_nd, checked_combos):
            continue
        # print(checked_combos)

        chain_rrt_star(new_pose, nearest_nd)

        # new_pose.parent = nearest_nd

        current_pos = new_pose

        moves.append(new_pose)

        # rewire_rrt_star_with_check(grid, moves, current_pos)
        counter += 1

        if goal[0] == new_pose.x and goal[1] == new_pose.y:
            path = []
            current_node = new_pose

            while current_node is not None:
                path.append((current_node.y, current_node.x))
                current_node = current_node.parent
            path.append((start[0], start[1]))
            path.reverse()

            return counter, path

    return None


def frrt_star():
    pass
