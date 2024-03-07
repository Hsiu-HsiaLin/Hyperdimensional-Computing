from collections import defaultdict

import numpy as np
from sklearn.cluster import KMeans

NUM_CLUSTERS = 4
INPUT_PATH = 'temp/similarities.txt'
OUTPUT_PATH = 'temp/clusters.txt'


def load_dataset(path):
    with open(path, 'r') as f:
        data = f.read()
    data = list(map(float, data.split(',')))
    return np.array(data)


def main():
    similarities = load_dataset(INPUT_PATH)

    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=0)
    kmeans.fit(similarities.reshape(-1, 1))

    centers = [center[0] for center in kmeans.cluster_centers_]

    for i in range(len(similarities)):
        print(f'{similarities[i]}: ({centers[kmeans.labels_[i]]})')

    center_to_points = defaultdict(list)
    for i in range(len(similarities)):
        center_to_points[centers[kmeans.labels_[i]]].append(similarities[i])

    for center, points in center_to_points.items():
        center_to_points[center] = sorted(points)

    point_to_center = {}
    for center, points in center_to_points.items():
        for point in points:
            point_to_center[point] = center

    clusters = defaultdict(list)
    for point in similarities:
        center = point_to_center[point]
        cluster_min = center_to_points[center][0]
        cluster_max = center_to_points[center][-1]

        normalized_similarity = (point - cluster_min) / (cluster_max - cluster_min)
        clusters[center].append(normalized_similarity)

    assert len(clusters.keys()) == NUM_CLUSTERS

    with open(OUTPUT_PATH, 'w') as f:
        for center, normalized_points in clusters.items():
            f.write(f'{center}#' + ','.join(map(str, normalized_points)) + '\n')


if __name__ == '__main__':
    main()
