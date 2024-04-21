from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

# Generate sample 2D LiDAR data (replace this with your actual data)
np.random.seed(0)
lidar_data = np.random.rand(100, 2) * 10

# Define DBSCAN parameters
epsilon = 0.5  # Neighborhood distance
min_samples = 1  # Minimum number of points in neighborhood to form a cluster

# Perform DBSCAN clustering
dbscan = DBSCAN(eps=epsilon, min_samples=min_samples)
clusters = dbscan.fit_predict(lidar_data)

# Filter out points labeled as noise (-1) or outliers
filtered_data = lidar_data[clusters != -1]

# Plot the original data
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(lidar_data[:, 0], lidar_data[:, 1], c='b', s=50, label='Original Data')
plt.title('Original LiDAR Data')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Plot the filtered data if it's not empty
if filtered_data.size > 0:
    plt.subplot(1, 2, 2)
    plt.scatter(filtered_data[:, 0], filtered_data[:, 1], c='r', s=50, label='Filtered Data')
    plt.title('Filtered LiDAR Data')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
else:
    plt.subplot(1, 2, 2)
    plt.title('Filtered LiDAR Data (No points)')
    plt.axis('off')

plt.tight_layout()

# Add legend with epsilon and min_samples values
plt.legend(title='DBSCAN Parameters', loc='upper left', fontsize='medium')
plt.gca().get_legend().get_title().set_fontsize('12')
plt.gca().get_legend().get_title().set_fontweight('bold')
plt.gca().get_legend().get_title().set_color('black')
plt.gca().get_legend().set_bbox_to_anchor((1, 1))

# Show plot
plt.show()
