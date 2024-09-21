import random
import os
from collections import namedtuple
from datetime import datetime
from math import cos, sin, pi, floor, isinf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap, BoundaryNorm

from sklearn.cluster import DBSCAN

from mapstate_ import MapState
from astar_ import astar
from rrt_ import rrt, rrt_connect, rrt_star, rrt_div, rrt_fast
from rrt_ import eliminate_duplicates, calculate_number_of_turns, calculate_smoothness

from util_ import bresenham_line


def plot_comparison_individual(test_results, total_tests):
    criteria = ['Number of Nodes', 'Execution Time [ms]', 'Path Length [m]',
                'Number of turns', 'Smoothness', 'Failure Rate (%)']
    num_criteria = len(criteria)

    for method, stats in test_results.items():
        data = [
            np.nanmean(stats['nodes']) if stats['nodes'] else 0,
            np.nanmean(np.array(stats['time']) / 1e6) if stats['time'] else 0,
            np.nanmean(stats['distance']) if stats['distance'] else 0,
            np.nanmean(stats['turns']) if 'turns' in stats and stats['turns'] else 0,
            np.nanmean(stats['smoothness']) if 'smoothness' in stats and stats['smoothness'] else 0,
            (stats['fails'] / total_tests) * 100
        ]

        fig, ax = plt.subplots(figsize=(8, 6))
        bar_width = 0.5
        index = np.arange(num_criteria)

        colors_bars = [
            "#00008B",
            "#FF8C00",
            "#8B0000",
            "#9932CC",
            "#8B4513",
            "#2F4F4F"
        ]
        bars = ax.bar(index, data, bar_width, color=colors_bars)

        ax.set_xlabel('Criteria')
        ax.set_ylabel('Values')
        ax.set_title(f'Performance of {method} by Various Criteria')
        ax.set_xticks(index)
        ax.set_xticklabels(criteria)

        plt.tight_layout()
        plt.show()


def write_comparison_to_excel(test_results, file_path, file_name, criteria, total_tests):
    methods = list(test_results.keys())
    data_frames = []

    for method in methods:
        data = {
            'Criteria': criteria,
            'Values': [
                np.nanmean(test_results[method]['nodes']) if test_results[method]['nodes'] else 0,
                np.nanmean(np.array(test_results[method]['time']) / 1e6) if test_results[method]['time'] else 0,
                np.nanmean(test_results[method]['distance']) if test_results[method]['distance'] else 0,
                np.nanmean(test_results[method]['turns']) if 'turns' in test_results[method] and test_results[method][
                    'turns'] else 0,
                np.nanmean(test_results[method]['smoothness']) if 'smoothness' in test_results[method] and
                                                                  test_results[method]['smoothness'] else 0,
                (test_results[method]['fails'] / total_tests * 100) if test_results[method]['fails'] else 0
            ]
        }
        df = pd.DataFrame(data)
        df['Method'] = method
        data_frames.append(df)

    result_df = pd.concat(data_frames, ignore_index=True)
    result_df.set_index(['Method', 'Criteria'], inplace=True)

    full_path = os.path.join(file_path, file_name)
    result_df.to_excel(full_path, engine='openpyxl')
    print(f"Data successfully written to {full_path}.")


def plot_comparison(test_results, total_tests):
    methods = list(test_results.keys())
    num_methods = len(methods)
    criteria = ['Number of Nodes', 'Execution Time [ms]', 'Path Length [m]',
                'Number of Turns', 'Smoothness', 'Failure Rate (%)']
    num_criteria = len(criteria)

    data = []
    for i in range(num_criteria):
        data.append([])

    for method in methods:
        stats = test_results[method]
        data[0].append(np.nanmean(stats['nodes']) if stats['nodes'] else 0)
        data[1].append(np.nanmean(np.array(stats['time']) / 1e6) if stats['time'] else 0)
        data[2].append(np.nanmean(stats['distance']) if stats['distance'] else 0)
        data[3].append(np.nanmean(stats['turns']) if stats['turns'] else 0)
        data[4].append(np.nanmean(stats['smoothness']) if stats['smoothness'] else 0)
        data[5].append((stats['fails'] / total_tests) * 100)

    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.15
    index = np.arange(num_methods)

    colors_bars = [
        "#00008B",
        "#FF8C00",
        "#8B0000",
        "#9932CC",
        "#8B4513",
        "#2F4F4F"
    ]
    for i in range(num_criteria):
        ax.bar(index + i * bar_width, data[i], bar_width, label=criteria[i], color=colors_bars[i])

    ax.set_xlabel('Methods')
    ax.set_ylabel('Values')
    ax.set_title('Comparison of Methods by Various Criteria')
    ax.set_xticks(index + bar_width * (num_criteria - 1) / 2)
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.xaxis.set_tick_params(labelsize=12)
    ax.legend()

    plt.tight_layout()
    plt.show()


def cache_fct(minimum, step, maximum, fct):
    dictFct = dict()
    while minimum <= maximum:
        val = minimum
        dictFct[val] = fct(val)
        minimum = minimum + step
    return dictFct


def place_goal(extended_matrix, goal):
    extended_matrix[*goal] = MapState.GOAL.value


def populate_map(lidar_x_data, lidar_y_data, robot, goal=None, resolution=0.25):
    def closest_integer(num):
        if num < 0:
            return int(num) - 1
        elif num > 0:
            return int(num) + 1
        else:
            return 0

    def x_discrete(el_to_disc):
        return int((el_to_disc - x_min) / resolution)

    def y_discrete(el_to_disc):
        return int((el_to_disc - y_min) / resolution)

    non_inf_numbers = [num for num in lidar_x_data if not np.isinf(num) and not np.isnan(num)]
    x_min = closest_integer(min(non_inf_numbers))
    non_inf_numbers = [num for num in lidar_x_data if not np.isinf(num) and not np.isnan(num)]
    x_max = closest_integer(max(non_inf_numbers))
    non_inf_numbers = [num for num in lidar_y_data if not np.isinf(num) and not np.isnan(num)]
    y_min = closest_integer(min(non_inf_numbers))
    non_inf_numbers = [num for num in lidar_y_data if not np.isinf(num) and not np.isnan(num)]
    y_max = closest_integer(max(non_inf_numbers))

    shape_col = (floor(int(y_max - y_min) / resolution)) + 1
    shape_lines = (floor(int(x_max - x_min) / resolution)) + 1
    matrix = np.full((shape_lines, shape_col), MapState.FREE.value, dtype=int)
    rows, cols = matrix.shape

    debug_print = False

    for i, _ in enumerate(lidar_x_data):
        if not isinf(lidar_x_data[i]) and not isinf(lidar_y_data[i]):
            lidar_x = int((lidar_x_data[i] - x_min) / resolution)
            lidar_y = int((lidar_y_data[i] - y_min) / resolution)
            if shape_lines > lidar_x and shape_col > lidar_y:
                matrix[lidar_x, lidar_y] = MapState.OBSTACLE.value

    onePointReduction = True
    point = [x_discrete(robot.x), y_discrete(robot.y)]

    if not onePointReduction:
        line1 = bresenham_line(x_discrete(robot.corner1[0]), y_discrete(robot.corner1[1]),
                               x_discrete(robot.corner2[0]), y_discrete(robot.corner2[1]))
        line3 = bresenham_line(x_discrete(robot.corner3[0]), y_discrete(robot.corner3[1]),
                               x_discrete(robot.corner4[0]), y_discrete(robot.corner4[1]))
        lines = []

        for p1 in line1:
            for p2 in line3:
                loc_line = bresenham_line(p1[0], p1[1], p2[0], p2[1])
                if loc_line:
                    lines.append(loc_line)

        for line in lines:
            for point_loc in line:
                matrix[point_loc[0], point_loc[1]] = MapState.ROBOT.value
    else:
        need_to_clean = True
        if need_to_clean:
            cleaning_space = int(robot.diag / resolution)
            offset = [cleaning_space - i for i in range(cleaning_space * 2 + 1)]
            neighborhood = [(i, j) for i in offset for j in offset]

            extended_matrix = matrix.copy()

            for dr, dc in neighborhood:
                new_row, new_col = point[0] + dr, point[1] + dc
                if (0 <= new_row < rows and 0 <= new_col < cols
                        and extended_matrix[new_row, new_col] != MapState.FREE.value):
                    extended_matrix[new_row, new_col] = MapState.FREE.value

            matrix = extended_matrix.copy()

        matrix[point[0], point[1]] = MapState.ROBOT.value

    num_blocks = int(robot.diag / resolution)

    extended_matrix = matrix.copy()

    extend_objects = True
    if extend_objects:
        offset = [num_blocks - i for i in range(num_blocks * 2 + 1)]
        neighborhood = [(i, j) for i in offset for j in offset]

        for row in range(rows):
            for col in range(cols):
                if extended_matrix[row, col] == MapState.OBSTACLE.value:
                    for dr, dc in neighborhood:
                        new_row, new_col = row + dr, col + dc
                        if 0 <= new_row < rows and 0 <= new_col < cols:
                            if matrix[new_row, new_col] == MapState.FREE.value:
                                extended_matrix[new_row, new_col] = MapState.EXTENDED_OBSTACLE.value

    # Region: goal
    if goal:
        extended_matrix[goal[0], goal[1]] = MapState.GOAL.value

    if debug_print:
        print("var = [", end="")
        for ind_b, line in  enumerate(extended_matrix):
            print("[", end="")
            for ind, el in enumerate(line):
                end = ", " if ind != len(line)-1 else "], \n"
                print(f"{el}", end=end)
        print("]")

    return point, extended_matrix


def plot_distance():
    with open("Distance_record.txt", 'r') as f:
        data = f.readlines()
        header_data = data[0][:-1].split(',')
        dict_index = dict()

        for dt in header_data:
            dict_index[dt] = header_data.index(dt)

        Ts = 1 / 60
        plt.figure()

        for line in data[1:]:
            data_line = line.split(',')
            distance_vector = [data_line[dict_index['field.distanceA']], data_line[dict_index['field.distanceB']],
                               data_line[dict_index['field.distanceC']], data_line[dict_index['field.distanceD']],
                               data_line[dict_index['field.distanceE']], data_line[dict_index['field.distanceF']]]
            plt.scatter(range(1, 7), distance_vector, color='b', marker='o')
            plt.xlim(0.5, 6.5)
            plt.ylim(0, 10)
            plt.xlabel('Sensor Number')
            plt.ylabel('Distance (m)')
            plt.title('Ultrasonic Distance Sensor Readings')
            plt.draw()
            plt.grid()
            plt.pause(Ts)
            plt.clf()


def apply_transformation(x, y, transformation_matrix):
    points = np.vstack((x, y, np.ones_like(x)))

    transformed_points = np.dot(transformation_matrix, points)

    transformed_x = transformed_points[0]
    transformed_y = transformed_points[1]

    return transformed_x, transformed_y


def rotation_matrix(angle, x_offset, y_offset):
    cos_theta = np.cos(angle)
    sin_theta = np.sin(angle)

    rotation_matrix_lines = np.array([[cos_theta, -sin_theta, x_offset],
                                      [sin_theta, cos_theta, y_offset],
                                      [0, 0, 1]])

    return rotation_matrix_lines


def calibrate_lidar1(x, y):
    angle = pi / 4
    x_off = -0.705 / 2
    y_off = -0.576 / 2

    matrix = rotation_matrix(angle, x_off, y_off)
    return apply_transformation(x, y, matrix)


def calibrate_lidar2(x, y):
    angle = -pi / 4 * 3
    x_off = 0.705 / 2
    y_off = 0.576 / 2

    matrix = rotation_matrix(angle, x_off, y_off)
    return apply_transformation(x, y, matrix)


def not_nan(el):
    return ~(np.isnan(el) & ~np.isinf(el))


def not_inf(el):
    return ~np.isinf(el)


def filter_fct(arr1, arr2, criteria=not_nan):
    valid_indices = criteria(arr1) & criteria(arr2)
    return arr1[valid_indices], arr2[valid_indices]


def filter_arr(arr, criteria=not_inf):
    return arr[criteria(arr)]


def plot_complete_scan(path_scan1, path_scan2):
    def plot_path(path_local, color_local, label_local):
        path_local = np.array(path_local)
        plt.plot(path_local[:, 1], path_local[:, 0], color=color_local,
                 marker='o', linestyle='-', label=label_local)
        legend_handles.append(plt.Line2D([0], [0], color=color_local,
                                         marker='o', linestyle='-', label=label_local))

    def get_item(key, array):
        idx_get_item = dict_scan[key]
        return array[:-1].split(',')[idx_get_item]

    def get_grid_type():
        return "RealTimeTestRandomGoal" if random_goal else "RealTimeTestFixedGoal"

    with open(path_scan1, 'r') as f:
        data_lidar1 = f.readlines()

    with open(path_scan2, 'r') as f:
        data_lidar2 = f.readlines()

    Robot = namedtuple('Robot', ['corner1', 'corner2', 'corner3', 'corner4',
                                 'x', 'y', 'angle', 'diag', 'x_disc', 'y_disc'])

    header_data_lidar1 = data_lidar1[0][:-1].split(',')

    start_ranges_offset = header_data_lidar1.index("field.ranges0")
    start_intensity_offset = header_data_lidar1.index("field.intensities0")

    nbOfRanges = start_intensity_offset - start_ranges_offset

    dict_scan = dict()

    for index, hd in enumerate(header_data_lidar1):
        dict_scan[hd] = index
        pass

    angle_min = float(get_item('field.angle_min', data_lidar1[1]))
    angle_max = float(get_item('field.angle_max', data_lidar1[1]))
    step = (angle_max - angle_min) / nbOfRanges
    angles = []
    cos_list = []
    sin_list = []

    while angle_min <= angle_max:
        val = angle_min
        angles.append(val)
        cos_list.append(cos(val))
        sin_list.append(sin(val))
        angle_min = angle_min + step

    center_x = 0
    center_y = 0
    robot_length = 0.705
    robot_width = 0.576

    robot = Robot(corner1=(-robot_length / 2, -robot_width / 2), corner2=(robot_length / 2, -robot_width / 2),
                  corner3=(robot_length / 2, robot_width / 2), corner4=(-robot_length / 2, robot_width / 2),
                  x=0, y=0, angle=0, diag=0.4301162633521313, x_disc=0, y_disc=0)

    x_min = 0
    x_max = 0
    y_min = 0
    y_max = 0

    tests = len(data_lidar1) - 3
    nb_of_tests = min(len(data_lidar1) - 3, tests)

    test_results = {
        'RRT': {'paths': [], 'nodes': [], 'time': [], 'distance': [], 'turns': [], 'smoothness': [], 'fails': 0},
        'RRT Connect': {'paths': [], 'nodes': [], 'time': [], 'distance': [], 'turns': [], 'smoothness': [],
                        'fails': 0},
        'RRT Div': {'paths': [], 'nodes': [], 'time': [], 'distance': [], 'turns': [], 'smoothness': [], 'fails': 0},
        'RRT Fast': {'paths': [], 'nodes': [], 'time': [], 'distance': [], 'turns': [], 'smoothness': [], 'fails': 0},
        'RRT Star': {'paths': [], 'nodes': [], 'time': [], 'distance': [], 'turns': [], 'smoothness': [], 'fails': 0}
    }
    criteria = ['Number of Nodes', 'Execution Time [ms]', 'Path Length [m]', 'Number of Turns', 'Smoothness',
                'Percentage of Failed Tests']

    algorithm_colors = {
        'A*': 'lime',
        'RRT': 'magenta',
        'RRT_Connect': (10 / 255, 155 / 255, 255 / 255),
        'RRT_Star': 'blue',
        'RRT_Div': 'red',
        'RRT_Fast': 'purple'
    }

    algorithm_labels = {
        'A*': 'A* Path',
        'RRT': 'RRT Path',
        'RRT_Connect': 'RRT-Connect Path',
        'RRT_Div': 'RRT-Div Path',
        'RRT_Fast': "FRRT Path",
        'RRT_Star': 'RRT* Path'
    }

    max_time = None
    lim_iter = 100_000_000
    random_goal = True

    for i in range(1, nb_of_tests + 1):
        print(f"Current iteration: {i}/{nb_of_tests} [{i / nb_of_tests * 100}%]")
        Ts = (float(get_item('%time', data_lidar1[i + 1])) - float(get_item('%time', data_lidar1[i]))) / 1_000_000_000

        dists_lidar1 = [float(r) for r in
                        data_lidar1[i][:-1].split(',')[start_ranges_offset:start_ranges_offset + nbOfRanges]]
        dists_lidar2 = [float(r) for r in
                        data_lidar2[i][:-1].split(',')[start_ranges_offset:start_ranges_offset + nbOfRanges]]

        intensity_lidar1 = [float(r) for r in
                            data_lidar1[i][:-1].split(',')[start_intensity_offset:start_intensity_offset + nbOfRanges]]
        intensity_lidar2 = [float(r) for r in
                            data_lidar2[i][:-1].split(',')[start_intensity_offset:start_intensity_offset + nbOfRanges]]

        x_lidar1 = np.array([center_x + dists_lidar1[idx_x] * cos_list[idx_x] for idx_x in range(nbOfRanges)])
        y_lidar1 = np.array([center_y + dists_lidar1[idx_y] * sin_list[idx_y] for idx_y in range(nbOfRanges)])

        x_lidar2 = np.array([center_x + dists_lidar2[idx_x] * cos_list[idx_x] for idx_x in range(nbOfRanges)])
        y_lidar2 = np.array([center_y + dists_lidar2[idx_y] * sin_list[idx_y] for idx_y in range(nbOfRanges)])

        filter_std = False
        filter_dbscan = True

        if sum(intensity_lidar1) != 0 and filter_std:
            mean_intensity = np.mean(intensity_lidar1)
            std_intensity = np.std(intensity_lidar1)
            threshold = mean_intensity + 2 * std_intensity
            index = np.greater(intensity_lidar1, threshold)
            x_lidar1 = np.array(x_lidar1)[index]
            y_lidar1 = np.array(y_lidar1)[index]

        if sum(intensity_lidar2) != 0 and filter_std:
            mean_intensity = np.mean(intensity_lidar2)
            std_intensity = np.std(intensity_lidar2)
            threshold = mean_intensity + 2 * std_intensity
            index = np.greater(intensity_lidar2, threshold)
            x_lidar2 = np.array(x_lidar2)[index]
            y_lidar2 = np.array(y_lidar2)[index]

        x_lidar1, y_lidar1 = calibrate_lidar1(x_lidar1, y_lidar1)
        x_lidar2, y_lidar2 = calibrate_lidar2(x_lidar2, y_lidar2)

        x_lidar1, y_lidar1 = filter_fct(x_lidar1, y_lidar1)
        x_lidar2, y_lidar2 = filter_fct(x_lidar2, y_lidar2)

        x_lidar = np.append(x_lidar1, x_lidar2)
        y_lidar = np.append(y_lidar1, y_lidar2)

        epsilon = 0.05
        min_samples = 5

        lidar_data1 = np.column_stack((x_lidar1, y_lidar1))
        lidar_data2 = np.column_stack((x_lidar2, y_lidar2))

        # lidar_data1 = lidar_data1[(~(np.isnan(lidar_data1) & ~np.isinf(lidar_data1))).any(axis=1)]

        if filter_dbscan:
            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(lidar_data1)

            lidar_data1 = lidar_data1[clusters != -1]

            x_lidar1 = lidar_data1[:, 0]
            y_lidar1 = lidar_data1[:, 1]

            # lidar_data2 = lidar_data2[(~(np.isnan(lidar_data2) & ~np.isinf(lidar_data2))).any(axis=1)]

            dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
            clusters = dbscan.fit_predict(lidar_data2)

            lidar_data2 = lidar_data2[clusters != -1]

        x_lidar2 = lidar_data2[:, 0]
        y_lidar2 = lidar_data2[:, 1]

        plot_one_filter = False
        if plot_one_filter:
            lidar_data = np.column_stack((x_lidar, y_lidar))
            lidar_data = lidar_data[~np.isnan(lidar_data).any(axis=1)]
            filtered_data = []

            if filter_dbscan:
                dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
                clusters = dbscan.fit_predict(lidar_data)

                filtered_data = lidar_data[clusters != -1]

            plt.figure(figsize=(12, 6))

            plt.subplot(1, 2, 1)
            plt.scatter(lidar_data[:, 0], lidar_data[:, 1], c='b', s=2)
            plt.plot([robot.corner1[0], robot.corner2[0], robot.corner3[0], robot.corner4[0]],
                     [robot.corner1[1], robot.corner2[1], robot.corner3[1], robot.corner4[1]], 'b-')
            plt.fill([robot.corner1[0], robot.corner2[0], robot.corner3[0], robot.corner4[0]],
                     [robot.corner1[1], robot.corner2[1], robot.corner3[1], robot.corner4[1]], color='lightblue')
            plt.title('Original LiDAR Data')
            plt.xlabel('X')
            plt.ylabel('Y')
            plt.grid(True)

            if filtered_data.size > 0:
                plt.subplot(1, 2, 2)
                plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c='r', s=2)
                plt.plot([robot.corner1[0], robot.corner2[0], robot.corner3[0], robot.corner4[0]],
                         [robot.corner1[1], robot.corner2[1], robot.corner3[1], robot.corner4[1]], 'b-')
                plt.fill([robot.corner1[0], robot.corner2[0], robot.corner3[0], robot.corner4[0]],
                         [robot.corner1[1], robot.corner2[1], robot.corner3[1], robot.corner4[1]], color='lightblue')
                plt.title('Filtered LiDAR Data')
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.grid(True)
            else:
                plt.subplot(1, 2, 2)
                plt.title('Filtered LiDAR Data (No points)')
                plt.axis('off')

            legend_handles = [plt.Rectangle((0, 0), 1, 1, color='white') for _ in range(2)]
            legend_labels = [f"Epsilon: {epsilon}", f"Minimum samples: {min_samples}"]
            plt.legend(legend_handles, legend_labels, loc='upper right', fontsize="200")

            plt.tight_layout()

        continuous_time = False
        if continuous_time:
            plt.plot([robot.corner1[0], robot.corner2[0], robot.corner3[0], robot.corner4[0]],
                     [robot.corner1[1], robot.corner2[1], robot.corner3[1], robot.corner4[1]], 'b-')
            plt.fill([robot.corner1[0], robot.corner2[0], robot.corner3[0], robot.corner4[0]],
                     [robot.corner1[1], robot.corner2[1], robot.corner3[1], robot.corner4[1]], color='lightblue')
            plt.plot(robot.corner1[0], robot.corner1[1], 'co', label='LIDAR1')
            plt.plot(robot.corner3[0], robot.corner3[1], 'ro', label='LIDAR2')

            plt.scatter(x_lidar1, y_lidar1, s=1, color='b', label='LIDAR1 Data')
            plt.scatter(x_lidar2, y_lidar2, s=1, color='r', label='LIDAR2 Data')

            plt.legend()
            plt.title("Sensor fusion visualization")
            plt.xlabel("Workspace X [m]")
            plt.ylabel("Workspace Y [m]")

            x_min = min([x_min, min(filter_arr(x_lidar))])
            x_max = max([x_max, max(filter_arr(x_lidar))])
            y_min = min([y_min, min(filter_arr(y_lidar))])
            y_max = max([y_max, max(filter_arr(y_lidar))])

        if not continuous_time:
            res = 0.25
            robot_point, map_matrix = populate_map(lidar_x_data=x_lidar, lidar_y_data=y_lidar,
                                                   robot=robot, resolution=res)
            y_loc, x_loc = map_matrix.shape
            x_min = 0
            y_min = 0
            x_max = max(x_max, x_loc)
            y_max = max(y_max, y_loc)
            rows = len(map_matrix)
            cols = len(map_matrix[0])

            max_time = Ts * 1_000_000_000

            goal_place = True

            if goal_place:
                if not random_goal:
                    goal = [min(rows - 1, 10), min(cols - 1, 40)]
                else:
                    goal = [random.randint(0, rows - 1), random.randint(0, cols - 1)]

                while map_matrix[goal[0]][goal[1]] != MapState.FREE.value:
                    goal = [random.randint(0, rows - 1), random.randint(0, cols - 1)]

                place_goal(extended_matrix=map_matrix, goal=goal)

            plt.imshow(map_matrix, cmap=map_cmap, norm=map_norm, interpolation='none')
            legend_labels = [state.name for state in MapState]
            legend_handles = [plt.Rectangle((0, 0), 1, 1,
                                            color=map_cmap(map_norm(i + 1))) for i in range(len(legend_labels))]

            plt.title('Results')
            plt.xlabel(f'X [grid tile size {res}m]')
            plt.ylabel(f'Y [grid tile size {res}m]')
            plt.xlim(x_min, x_max)
            plt.ylim(y_min, y_max)

            # algorithms = ['RRT', 'RRT_Connect', 'RRT_Div', 'RRT_Fast', 'RRT_Star']
            algorithms = []
            for algorithm in algorithms:
                result = None

                if algorithm == 'A*':
                    result = astar(map_matrix, (robot_point[0], robot_point[1]), [goal[1], goal[0]], lim_iter, max_time)
                elif algorithm == 'RRT':
                    result = rrt(map_matrix, (robot_point[0], robot_point[1]), [goal[0], goal[1]], lim_iter, max_time)
                elif algorithm == 'RRT_Connect':
                    result = rrt_connect(map_matrix, (robot_point[0], robot_point[1]), [goal[0], goal[1]],
                                         lim_iter, max_time)
                elif algorithm == 'RRT_Star':
                    result = rrt_star(map_matrix, (robot_point[0], robot_point[1]), [goal[0], goal[1]],
                                      lim_iter, max_time)
                elif algorithm == 'RRT_Div':
                    result = rrt_div(map_matrix, (robot_point[0], robot_point[1]), [goal[0], goal[1]],
                                     lim_iter, max_time)
                elif algorithm == 'RRT_Fast':
                    result = rrt_fast(map_matrix, (robot_point[0], robot_point[1]), [goal[0], goal[1]],
                                      lim_iter, max_time)

                if result:
                    nb_of_nodes, path, dt, dist = result

                    num_turns = calculate_number_of_turns(eliminate_duplicates(path))
                    smoothness_value = calculate_smoothness(eliminate_duplicates(path))

                    test_results[algorithm.replace('_', ' ')]['paths'].append(path)
                    test_results[algorithm.replace('_', ' ')]['nodes'].append(nb_of_nodes)
                    test_results[algorithm.replace('_', ' ')]['time'].append(dt)
                    test_results[algorithm.replace('_', ' ')]['distance'].append(dist)
                    test_results[algorithm.replace('_', ' ')]['turns'].append(num_turns)
                    test_results[algorithm.replace('_', ' ')]['smoothness'].append(smoothness_value)

                    path = np.array(path)
                    plt.plot(path[:, 1], path[:, 0],
                             color=algorithm_colors[algorithm], marker='o')

                    legend_labels.append(algorithm_labels[algorithm])
                    legend_handles.append(
                        plt.Line2D([0], [0], marker='o', color='w',
                                   markerfacecolor=algorithm_colors[algorithm],
                                   markersize=10, label=algorithm_labels[algorithm]))

                    print(f"{algorithm} Solution found with {nb_of_nodes} searches in {dt / 1e9:.3f} seconds")
                else:
                    test_results[algorithm.replace('_', ' ')]['fails'] += 1
                    print(f"{algorithm} Solution isn't found")

            plt.legend(legend_handles, legend_labels, loc='upper right')

        # print("")
        # print(Ts)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.draw()
        plt.grid()
        plt.tight_layout()
        plt.pause(Ts)
        plt.clf()

    plot_comparison_individual(test_results, nb_of_tests)
    plot_comparison(test_results, nb_of_tests)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'result_{get_grid_type()}_{current_time}.xlsx'
    filepath = "./results"
    write_comparison_to_excel(test_results, filepath, filename, criteria, nb_of_tests)


def plot_scan(*paths, nbOfRanges=1681):
    # def draw_binary_frame():
    #     dists = [float(r) for r in line[:-1].split(',')[start_ranges_offset:start_ranges_offset + nbOfRanges]]
    #
    #     x = [center_x + dists[idx_x] * cos_list[idx_x] for idx_x in range(nbOfRanges)]
    #     y = [center_y + dists[idx_y] * sin_list[idx_y] for idx_y in range(nbOfRanges)]
    #
    #     occupancy_matrix = populate_map(lidar_x_data=x, lidar_y_data=y, resolution=0.10)
    #
    #     plt.imshow(occupancy_matrix, cmap='binary', origin='lower')
    #
    #     plt.title('Occupancy Map')
    #     plt.xlabel('X')
    #     plt.ylabel('Y')
    #
    #     path_astar = astar(occupancy_matrix, (0, 0), (30, 25))
    #     if path_astar:
    #         path_astar = np.array(path_astar)
    #         plt.plot(path_astar[:, 1], path_astar[:, 0], color='lime', marker='o')

    def draw_frame():
        dists = [float(r) for r in line[:-1].split(',')[start_ranges_offset:start_ranges_offset + nbOfRanges]]

        x = [center_x + dists[idx_x] * cos_list[idx_x] for idx_x in range(nbOfRanges)]
        y = [center_y + dists[idx_y] * sin_list[idx_y] for idx_y in range(nbOfRanges)]

        ax.plot(0, 0, 'ro')
        ax.plot(x, y)
        ax.grid()

    def get_item(key, array):
        idx_get_item = dict_scan[key]
        return array[:-1].split(',')[idx_get_item]

    data = dict()
    printDebug = False

    for path in paths:
        with open(path, 'r') as f:
            data[path] = f.readlines()

    header_data = data[paths[0]][0][:-1].split(',')

    start_ranges_offset = header_data.index("field.ranges0")
    # start_intensity_offset = start_ranges_offset + nbOfRanges

    dict_scan = dict()

    for index, hd in enumerate(header_data):
        dict_scan[hd] = index
        pass

    angle_min = float(get_item('field.angle_min', data[paths[0]][1]))
    angle_max = float(get_item('field.angle_max', data[paths[0]][1]))
    step = (angle_max - angle_min) / nbOfRanges
    angles = []
    cos_list = []
    sin_list = []

    while angle_min <= angle_max:
        val = angle_min
        angles.append(val)
        cos_list.append(cos(val))
        sin_list.append(sin(val))
        angle_min = angle_min + step

    center_x = 0
    center_y = 0

    Ts = (float(get_item('%time', data[paths[0]][2])) - float(get_item('%time', data[paths[0]][1]))) / 1_000_000_000
    fig = plt.figure(figsize=(100, 100))
    nb = len(paths)

    row_nb = int(nb / 2) if nb % 2 == 0 else int(nb / 2) + 1
    col_nb = 1 if nb == 1 else 2

    draw_binary = False

    for i in range(1, len(data[paths[0]])):
        for idx, path in enumerate(paths):
            ax = fig.add_subplot(row_nb, col_nb, idx + 1)
            ax.set_title(path)
            line = data[path][i]

            # if draw_binary:
            #     draw_binary_frame()
            # else:
            draw_frame()

        plt.tight_layout()
        plt.draw()
        plt.pause(Ts)
        plt.clf()

    if printDebug:
        print(get_item('field.range_min', data[1]))
        print(get_item('field.range_max', data[1]))

        print(header_data[start_ranges_offset])


def print_header(path):
    with open(path, 'r') as f:
        header = f.readline()
        for ind, hd in enumerate(header[:-1].split(',')):
            print(hd, " ", ind)


def main_local():
    final_Version = True

    if final_Version:
        plot_complete_scan(r'C:\Users\Iulian\Desktop\scan1_1_120.txt', r'C:\Users\Iulian\Desktop\scan2_1_120.txt')
    else:
        plot_complete_scan('lidarBackScan1.txt', 'lidarFrontScan2.txt')
        plot_scan(r'C:\Users\Iulian\Desktop\scan1_1_120.txt', r'C:\Users\Iulian\Desktop\scan2_1_120.txt')


def main_test():
    final_Version = True

    if final_Version:
        plot_scan('lidarFrontScan1.txt', 'lidarFrontScan2.txt', 'lidarBackScan1.txt', 'lidarBackScan2.txt')
    else:
        cache_sin = cache_fct(-pi * 10, 0.01, pi * 10, sin)
        cache_cos = cache_fct(-pi, 0.01, pi, cos)

        for sn in cache_sin.keys():
            print(sn)

        for cs in cache_cos.keys():
            print(cs)


if __name__ == '__main__':
    colors = ['blue', 'white', 'black', (0.6, 0.6, 0.6), 'green']
    map_cmap = ListedColormap(colors, name='color_map')
    bounds = [i + 1 for i in range(len(colors) + 1)]
    map_norm = BoundaryNorm(bounds, len(bounds) - 1)

    main_local()
