import random

import numpy as np

np.random.seed(0)
random.seed(0)


def rotate_cluster(points):
    transposed_points = np.transpose(points)
    rotated_points = np.flip(transposed_points, axis=0)
    return np.transpose(rotated_points)


def generate_locations(num_clusters):
    offsets = []
    for i in range(num_clusters):
        for j in range(num_clusters):
            offsets.append([2 * i, 2 * j])
    return offsets


def gen_moons(n_samples, centers, random_state=0):
    np.random.seed(random_state)
    random.seed(random_state)

    num_clusters = centers
    offsets = generate_locations(num_clusters)
    X, y = [], []
    for i in range(num_clusters):
        radius = np.random.uniform(0.5, 2)
        distance = np.random.uniform(0.5, 2) * radius
        noise = np.random.uniform(0.01, 0.3)

        t = np.random.uniform(np.pi, 3 * np.pi, n_samples)
        o = random.sample(offsets, 1)[0]
        offset_x = o[0] + radius * np.cos(t) + np.random.randn(n_samples) * noise
        offset_y = o[1] + radius * np.sin(t) - distance + np.random.randn(n_samples) * noise

        X_cluster = np.column_stack((offset_x, offset_y))

        min_x, min_y = np.min(X_cluster, axis=0)
        max_x, max_y = np.max(X_cluster, axis=0)

        idx = X_cluster[:, 1] > 0.5 * (max_y - min_y) + min_y
        X_cluster = X_cluster[idx, :]

        if np.random.choice([0, 1]) == 0:
            X_cluster = rotate_cluster(X_cluster)

        X.extend(X_cluster)
        y.extend([i] * np.sum(idx))

    # Convert lists to arrays
    X = np.array(X)
    y = np.array(y)

    return X, y
