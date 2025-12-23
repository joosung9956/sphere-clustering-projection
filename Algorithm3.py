import numpy as np
import matplotlib.pyplot as plt
import heapq

# 5D test data
np.random.seed(42)
cluster1 = np.random.normal(loc=(0, 0, 0, 0, 0), scale=1.0, size=(300, 5))
cluster2 = np.random.normal(loc=(0.3, 0.3, 5.0, 0.1, 0), scale=1.0, size=(200, 5))
points_5d = np.concatenate([cluster1, cluster2], axis=0)
n_points = len(points_5d)

# 2D projection function
def project_to_2d(point, base_point, normal):
    projected = point - np.dot(point - base_point, normal) * normal
    arb = np.zeros(5)
    arb[0] = 1.0
    if abs(np.dot(arb, normal)) > 0.99:
        arb[1] = 1.0
    u = arb - np.dot(arb, normal) * normal
    u /= np.linalg.norm(u) + 1e-8
    arb2 = np.zeros(5)
    arb2[2] = 1.0
    v = arb2 - np.dot(arb2, normal) * normal - np.dot(arb2, u) * u
    v /= np.linalg.norm(v) + 1e-8
    x = np.dot(projected - base_point, u)
    y = np.dot(projected - base_point, v)
    return np.array([x, y])

# Algorithm 3 with distance threshold
def algorithm3_with_threshold(points_5d, threshold=2.0, base_weight=5.0):
    working_points = points_5d.copy()
    n = len(working_points)
    idx = np.random.choice(n, 3, replace=False)
    base_point = np.mean(working_points[idx], axis=0)
    v1 = working_points[idx[1]] - working_points[idx[0]]
    v2 = working_points[idx[2]] - working_points[idx[0]]
    v1 /= np.linalg.norm(v1) + 1e-8
    v2 /= np.linalg.norm(v2) + 1e-8
    arbitrary = np.zeros(5)
    arbitrary[0] = 1.0
    normal = arbitrary - np.dot(arbitrary, v1) * v1 - np.dot(arbitrary, v2) * v2
    if np.linalg.norm(normal) < 1e-6:
        arbitrary[1] = 1.0
        normal = arbitrary - np.dot(arbitrary, v1) * v1 - np.dot(arbitrary, v2) * v2
    normal /= np.linalg.norm(normal) + 1e-8
    
    projected_2d = np.zeros((n, 2))
    for i in idx:
        projected_2d[i] = project_to_2d(working_points[i], base_point, normal)
    
    priority_queue = []
    for i in range(n):
        if i not in idx:
            dist = np.linalg.norm(points_5d[i] - base_point)
            heapq.heappush(priority_queue, (dist, i))
    
    while priority_queue:
        _, new_idx = heapq.heappop(priority_queue)
        new_point = points_5d[new_idx].copy()
        dist = np.linalg.norm(new_point - base_point)
        if dist <= threshold:
            projected_2d[new_idx] = project_to_2d(new_point, base_point, normal)
        else:
            direction = new_point - base_point
            direction /= np.linalg.norm(direction) + 1e-8
            weight = base_weight * (dist / threshold)
            weight = min(weight, base_weight * 3)
            normal += weight * direction
            normal /= np.linalg.norm(normal) + 1e-8
            projected_2d[new_idx] = project_to_2d(new_point, base_point, normal)
            base_point = working_points[new_idx].copy()
    
    return projected_2d

# Run and plot
optimal_2d = algorithm3_with_threshold(points_5d, threshold=7.0, base_weight=0)
labels = np.zeros(n_points)
labels[300:] = 1

plt.figure(figsize=(18, 6))

# 1. Original front view (X-Y): heavy overlap
plt.subplot(1, 3, 1)
plt.scatter(points_5d[:, 0], points_5d[:, 1], c=labels, cmap='coolwarm', s=15, alpha=0.7)
plt.title('Original front view (X-Y: heavy overlap)')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# 2. Side view (X-Z): clear separation
plt.subplot(1, 3, 2)
plt.scatter(points_5d[:, 0], points_5d[:, 2], c=labels, cmap='coolwarm', s=15, alpha=0.7)
plt.title('Side view (X-Z: clear separation)')
plt.xlabel('X')
plt.ylabel('Z')
plt.axis('equal')
plt.grid(True, alpha=0.3)

# 3. Algorithm 3 result
plt.subplot(1, 3, 3)
plt.scatter(optimal_2d[:, 0], optimal_2d[:, 1], c=labels, cmap='coolwarm', s=15, alpha=0.7)
plt.title('Algorithm 3 adaptive projection')
plt.xlabel('Projected U')
plt.ylabel('Projected V')
plt.axis('equal')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
