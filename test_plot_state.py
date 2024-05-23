import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, LinearSegmentedColormap
from enum import Enum


class State(Enum):
    ROBOT = 1
    FREE = 2
    OBSTACLE = 3
    EXTENDED_OBJECT = 4


def test_mat():
    rows = 5
    cols = 5

    matrix = np.full((rows, cols), State.FREE.value)

    matrix[1, 1] = State.ROBOT.value
    matrix[2, 2] = State.EXTENDED_OBJECT.value
    matrix[2, 1] = State.FREE.value
    matrix[3, 3] = State.OBSTACLE.value

    bounds = [1, 2, 3, 4, 5]  # Add one more value than the number of states
    norm = BoundaryNorm(bounds, len(bounds) - 1)

    colors = ['blue', 'white', 'orange', 'red']

    cmap = ListedColormap(colors)

    plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='none')

    legend_labels = [state.name for state in State]
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=cmap(norm(i+1)))
        for i in range(len(legend_labels))
    ]

    plt.legend(legend_handles, legend_labels, loc='upper right')
    plt.grid(color='black', linewidth=2)

    plt.title('State Matrix')
    plt.show()


def plot_std_example():
    lidar_data = np.random.normal(loc=100, scale=20, size=1000)

    lidar_data = (lidar_data - np.min(lidar_data)) / (np.max(lidar_data) - np.min(lidar_data))

    mean = np.mean(lidar_data)
    std_dev = np.std(lidar_data)

    lower_threshold = mean - 0.97 * std_dev
    upper_threshold = mean + 0.97 * std_dev

    colors = ["darkblue", "red", "darkblue"]
    n_bins = 100
    cmap_name = 'custom_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    fig, ax = plt.subplots(figsize=(10, 6))

    norm = plt.Normalize(lidar_data.min(), lidar_data.max())

    n, bins, patches = ax.hist(lidar_data, bins=30, alpha=0.7, label='LiDAR Intensity Data')

    for c, p in zip(bins, patches):
        plt.setp(p, 'facecolor', cm(norm(c)))

    plt.axvline(lower_threshold, color='green', linestyle='dashed', linewidth=2, label=f'Lower 3% = {lower_threshold:.2f}')
    plt.axvline(upper_threshold, color='green', linestyle='dashed', linewidth=2, label=f'Upper 3% = {upper_threshold:.2f}')
    plt.axvline(mean, color='orange', linestyle='dashed', linewidth=2, label=f'Mean = {mean:.2f}')

    plt.title('Standard Deviation of LiDAR Output Intensity')
    plt.xlabel('Normalized Intensity')
    plt.ylabel('Frequency')
    plt.legend()

    plt.show()

plot_std_example()
