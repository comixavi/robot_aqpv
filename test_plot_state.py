import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from enum import Enum


class State(Enum):
    ROBOT = 1
    FREE = 2
    OBSTACLE = 3
    EXTENDED_OBJECT = 4


# Define the dimensions of the matrix
rows = 5
cols = 5

# Initialize the matrix with the "FREE" state
matrix = np.full((rows, cols), State.FREE.value)

# Set some states to different values for demonstration
matrix[1, 1] = State.ROBOT.value
matrix[2, 2] = State.EXTENDED_OBJECT.value
matrix[2, 1] = State.FREE.value
matrix[3, 3] = State.OBSTACLE.value

# Define boundaries for each state
bounds = [1, 2, 3, 4, 5]  # Add one more value than the number of states
norm = BoundaryNorm(bounds, len(bounds) - 1)

# Define colors for each state
colors = ['blue', 'white', 'orange', 'red']

# Create a colormap
cmap = ListedColormap(colors)

# Plot the matrix
plt.imshow(matrix, cmap=cmap, norm=norm, interpolation='none')

# Create a legend
legend_labels = [state.name for state in State]
legend_handles = [
    plt.Rectangle((0, 0), 1, 1, color=cmap(norm(i+1)))
    for i in range(len(legend_labels))
]  # Set the color of ROBOT state to blue in legend

plt.legend(legend_handles, legend_labels, loc='upper right')
plt.grid(color='black', linewidth=2)

plt.title('State Matrix')
plt.show()
