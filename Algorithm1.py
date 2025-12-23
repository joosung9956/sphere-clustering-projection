import random
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # Required for 3D projection

# Algorithm1
def distance(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2 +
        (p1[2] - p2[2])**2
    )

def algorithm1(points, d=1.0):
    if len(points) == 0:
        return []
    
    n = len(points)
    clusters = []
    remaining = set(range(n))
    
    while remaining:
        # Pick random uncovered point as starting point
        start_point = random.choice(list(remaining))
        A = set()
        queue = [start_point]
        A.add(start_point)
        remaining.remove(start_point)
        
        while queue:
            new_batch = set()
            for current_point in queue:
                for remaining_point in list(remaining):
                    if distance(points[current_point], points[remaining_point]) <= d:
                        new_batch.add(remaining_point)
            
            for remaining_point in new_batch:
                remaining.remove(remaining_point)
                A.add(remaining_point)
            
            queue = list(new_batch)
        
        clusters.append(sorted(list(A)))  # Sort cluster indices
    
    return clusters

# Generate sample data
np.random.seed(42)
cluster1 = np.random.normal(loc=[0, 0, 0], scale=1.0, size=(300, 3))
cluster2 = np.random.normal(loc=[5, 5, 5], scale=1.0, size=(200, 3))
points = np.vstack([cluster1, cluster2]).tolist()  # List of [x, y, z]

# Run algorithm
clusters = algorithm1(points, d=2.0)

# Plot results
fig = plt.figure(figsize=(12, 5))

# Left: Before clustering (all points black)
ax1 = fig.add_subplot(121, projection='3d')
points_arr = np.array(points)
ax1.scatter(points_arr[:, 0], points_arr[:, 1], points_arr[:, 2], c='black', s=10)
ax1.set_title("Before Clustering\n(500 points, d=2.0)")
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Right: After clustering (colored by cluster)
ax2 = fig.add_subplot(122, projection='3d')
colors = plt.cm.tab10(np.linspace(0, 1, len(clusters)))  # One color per cluster

for i, cluster_idx in enumerate(clusters):
    cluster_points = points_arr[cluster_idx]
    ax2.scatter(cluster_points[:, 0], cluster_points[:, 1], cluster_points[:, 2],
                c=[colors[i]], label=f'Cluster {i} ({len(cluster_points)} points)', s=10)

ax2.set_title("Algorithm 1 Result (d=2.0)")
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.legend()

plt.tight_layout()
plt.show()
