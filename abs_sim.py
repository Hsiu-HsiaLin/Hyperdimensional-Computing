
GROUPS_PATH = 'data/groups.csv'
CLUSTERS_PATH = 'temp/clusters.txt'
OUTPUT_PATH = 'temp/abs_similarities.txt'


class Group:
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return f'({self.min_val}, {self.max_val})'


class Cluster:
    def __init__(self, similarities):
        self.similarities = similarities

    def __repr__(self):
        return f'Cluster: {self.similarities}'


def load_groups(path):  # Ground truth
    # should be sorted by value
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = lines[1:]  # remove header
    groups = []
    for line in lines:
        line = line.strip()
        values = line.split(',')
        assert len(values) == 2
        groups.append(Group(float(values[0]), float(values[1])))
    return groups


def load_clusters(path):
    with open(path, 'r') as f:
        lines = f.readlines()

    clusters = []
    for line in lines:
        line = line.strip()
        _, similarities = line.split('#')
        similarities = similarities.split(',')
        similarities = list(map(float, similarities))
        clusters.append(Cluster(similarities))

    return clusters


def main():
    ground_truth_groups = load_groups(GROUPS_PATH)
    clusters = load_clusters(CLUSTERS_PATH)
    abs_similarities = []

    for i, cluster in enumerate(clusters):
        group = ground_truth_groups[i]
        cluster_abs_similarities = []
        for similarity in cluster.similarities:
            abs_similarity = group.min_val + similarity * (group.max_val - group.min_val)
            cluster_abs_similarities.append(abs_similarity)

        abs_similarities.append(cluster_abs_similarities)

    with open(OUTPUT_PATH, 'w') as f:
        for cluster_abs_similarities in abs_similarities:
            f.write(','.join(map(str, cluster_abs_similarities)) + '\n')


if __name__ == '__main__':
    main()
