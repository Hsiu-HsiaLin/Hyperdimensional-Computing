import copy
import math
import os
from typing import List

import cv2
import numpy as np

DATASET_PATH = 'data/pic'
D = 10000  # dimensions in random space
NUM_LEVELS = 100

OUTPUT_PATH = 'temp/similarities.txt'


def load_dataset(path):
    dataset = []
    for folder in os.listdir(path):
        folder_path = os.path.join(path, folder)
        # only 2 images per folder
        images = [cv2.imread(os.path.join(folder_path, img), cv2.IMREAD_UNCHANGED) for img in os.listdir(folder_path)]
        assert len(images) == 2
        dataset.extend(images)
    return dataset


def get_levels(vectors, num_levels) -> List[float]:
    min_val = math.inf
    max_val = -math.inf
    levels = []
    for vector in vectors:
        min_val = min(min(vector), min_val)
        max_val = max(max(vector), max_val)
    length = max_val - min_val
    gap = length / num_levels
    for i in range(num_levels):
        levels.append(min_val + i * gap)
    levels.append(max_val)
    return levels


def generate_level_hvs(num_levels, d) -> List[np.ndarray]:  # num_levels * d (1 and -1)
    np.random.seed(0)

    level_hvs = []
    indexes = list(range(d))
    base_change = d // 2
    next_level_change = int(D/2/num_levels)

    base = np.full(d, -1)
    for i in range(num_levels):
        if i == 0:
            to_one = np.random.permutation(indexes)[:base_change]
        else:
            to_one = np.random.permutation(indexes)[:next_level_change]
        for index in to_one:
            base[index] *= -1
        level_hvs.append(copy.deepcopy(base))
    for hv in level_hvs:
        for i in range(len(hv)):
            if hv[i] == -1:
                hv[i] = 0
    return level_hvs


def generate_id_hvs(size, d) -> List[np.array]:
    np.random.seed(0)

    id_hvs = []
    indexes = list(range(d))
    for i in range(size):
        base = np.full(d, 0)
        to_one = np.random.permutation(indexes)[:d // 2]
        for index in to_one:
            base[index] = 1

        id_hvs.append(base)

    return id_hvs


def get_level(value, levels) -> int:
    if value == levels[-1]:
        return len(levels) - 2  # check
    upper_index = len(levels) - 1
    lower_index = 0

    key_index = 0
    while upper_index > lower_index:
        key_index = int((upper_index + lower_index)/2)
        if levels[key_index] <= value < levels[key_index + 1]:
            return key_index
        if levels[key_index] > value:
            upper_index = key_index
        else:
            lower_index = key_index
        key_index = int((upper_index + lower_index)/2)
    return key_index


def encode(vector, d, levels, level_hvs, id_hvs) -> np.array:
    # encode vector to hv
    hv = np.zeros(d, dtype=np.int64)
    for i, value in enumerate(vector):
        id_hv = id_hvs[i]
        level = get_level(value, levels)
        level_hv = level_hvs[level]
        hv += (id_hv ^ level_hv)
    return hv


def convert_to_similarity(hamming_distance, d) -> float:
    return 1 - (hamming_distance / d)


def main():
    dataset = load_dataset(DATASET_PATH)
    dataset = np.array(dataset)

    num_samples = dataset.shape[0]
    img_size = dataset.shape[1] * dataset.shape[2]

    dataset = dataset.reshape((num_samples, img_size))

    levels = get_levels(dataset, NUM_LEVELS)

    level_hvs = generate_level_hvs(NUM_LEVELS, D)
    id_hvs = generate_id_hvs(img_size, D)

    hvs = []
    for vector in dataset:
        hvs.append(encode(vector, D, levels, level_hvs, id_hvs))

    for hv in hvs:
        for i, val in enumerate(hv):
            hv[i] = 0 if val < img_size / 2 else 1

    hamming_distances = []
    for i in range(0, num_samples, 2):
        hamming_distance = np.sum(np.abs(hvs[i] - hvs[i + 1]))
        hamming_distances.append(hamming_distance)

    similarities = [convert_to_similarity(distance, D) for distance in hamming_distances]

    with open(OUTPUT_PATH, 'w') as f:
        f.write(','.join(map(str, similarities)) + '\n')


if __name__ == '__main__':
    main()
