import os

import time
import random
import joblib

import mat73 as mat
import numpy as np
import pandas as pd

import plotly.graph_objects as go

import umap
import sklearn

'''
mat 확장자로 저장된 data를 python으로 불러오는 함수
'''


def import_data(data_path: str):
    INTERACTION_COLUMN_IDX = 2  # nd column (2D in matrix)
    CALCIUM_COLUMN_START_INDEX = 6  # th column (2D in matrix)

    mat_data = mat.loadmat(data_path)
    interaction_data = mat_data['Datastorage'][:, INTERACTION_COLUMN_IDX - 1]
    calcium_data = mat_data['Datastorage'][:, (CALCIUM_COLUMN_START_INDEX - 1):]
    return interaction_data, calcium_data


'''
Interaction하는 bout만 추출하는 함수
'''


def extract_interaction(interaction_data: np.ndarray,
                        calcium_data: np.ndarray,
                        fs: float,
                        time_bin: float,
                        shuffle: bool = True):
    def is_all_equal(array: np.ndarray,
                     target: int):
        return all(element == target for element in array)

    # Output
    interaction_calcium_data = []

    # Variable
    i = 0

    # BIN_LENGTH_TO_INDEX indices need to make time_bin_ms bin.
    BIN_LENGTH_TO_INDEX = int(time_bin / (1000 / fs))

    while i + BIN_LENGTH_TO_INDEX < len(interaction_data):
        temp_interaction_block = interaction_data[i:i + BIN_LENGTH_TO_INDEX]
        temp_calcium_block = calcium_data[i:i + BIN_LENGTH_TO_INDEX, :]

        if is_all_equal(temp_interaction_block, 1) is True:
            # 1D: Interaction bout, 2D: Cell, 3D: Bin OR 1D: Interaction bout, 2D: Mean event of cell
            temp_calcium_block = np.average(temp_calcium_block, axis=0)  # Average bin
            interaction_calcium_data.append(np.transpose(temp_calcium_block))

            # Increase index
            i += BIN_LENGTH_TO_INDEX

        if is_all_equal(temp_interaction_block, 1) is False:
            # Increase index
            i += 1

    # Shuffle dataset randomly
    if shuffle is True:
        random.shuffle(interaction_calcium_data)

    return np.array(interaction_calcium_data)


'''
mat 확장자로 저장된 data를 불러오고, interaction하는 bout만 추출하는 함수
'''


def read_data(trial_label: list,
              data_path: str,
              fps: float,
              time_bin: float,
              shuffle=True):
    # Output
    data = {}

    for label, path in zip(trial_label, data_path):
        temp_interaction, temp_calcium = import_data(path)
        data[label] = extract_interaction(temp_interaction, temp_calcium, fps, time_bin, shuffle=shuffle)
    return data


'''
실험에 사용할 데이터만 선택하는 함수
'''


def select_data(data: dict, org: list):
    # Return
    using_data = {}

    # Reshape orgs
    org_keys = np.array(org).reshape(-1)

    for key in data.keys():
        if key in org_keys:
            using_data[key] = data[key]

    return using_data


'''
전체 data의 길이를 최소 data 길이로 맞추는 함수
길이를 맞추기 위해 최소 길이보다 긴 배열의 끝 (end)을 제거함
'''


def match_data_size(data: dict):
    # Find minimum interaction duration
    minimum_length = np.inf  # Set largest length
    minimum_data_key = ''  # Dictionary key for minimum interaction duration

    for key in data.keys():
        if len(data[key]) < minimum_length:
            minimum_length = len(data[key])
            minimum_data_key = key

    # Match dataset size
    for key in data.keys():
        if key != minimum_data_key:  # For larger data
            # Create temp dataset
            data_handle = data[key]
            temp_data = []

            # Pick random bout of interaction
            random_selection = np.random.randint(0, len(data_handle), size=minimum_length)

            # Pick data
            for bout in random_selection:
                temp_data.append(data_handle[bout])

            # Assign size-matched data
            data[key] = np.array(temp_data)
        if key == minimum_data_key:  # Do not change the smallest data
            pass

    return data


'''
Raw 데이터를 train data와 test data로 나누는 함수
'''


def split_data(data: dict, method: str = 'ratio', train_ratio: float = 0.8):
    # Result
    train_data, test_data = {}, {}

    # Split data
    if method == 'ratio':
        for key, item in data.items():
            TRAIN_LENGTH = int(len(item) * train_ratio)
            TRAIN_IDX = np.random.randint(low=0, high=len(item), size=TRAIN_LENGTH)
            TEST_IDX = np.delete(np.arange(len(item)), TRAIN_IDX)
            train_data[key] = item[TRAIN_IDX]
            test_data[key] = item[TEST_IDX]
    else:
        raise ValueError('Unexpected method.')

    return train_data, test_data


'''
Feature를 만들기 위한 iteration을 설정하는 함수 (Boyle, Fusi et al., 2023, Neuron)
'''


def set_iter(q: int, n: int):
    # Parameters
    # q [int]: Number of features to extract from each cell.
    # n [int]: Number of cells.
    # Returns
    # T [int]: Number to iterate.
    return 2 * q * n  # T


'''
Feature를 만들기 위해 모든 ensemble cell (neuron)에 대해 interaction bout 중 q만큼 random bout을 추출하는 함수 
(Boyle, Fusi et al., 2023, Neuron)
'''


def select_feature_from_random_bout(data: np.ndarray, cell_idx: int, q: int):
    # Reset returns
    features = []

    # Pick random bout of interaction for num_q times
    random_selection = np.random.randint(0, len(data), size=q)

    # Pick random features
    for bout in random_selection:
        features.append(data[bout][cell_idx])

    return features


'''
Feature를 추출하는 함수
'''


def extract_features(data: dict, q: int):
    # Check validity
    if not isinstance(data, dict):
        raise ValueError('Given social is not a <dict>. Check the data type.')

    # Return
    feature_data: dict = {}  # Dict

    # Feature selection
    for key in data.keys():
        temp_data_handle = data[key]  # Data handle
        temp_data_output = []  # Result of feature selection from social data handle

        # Set iter; T = 2*q*n
        T = set_iter(q, len(temp_data_handle[0]))

        for iteration in range(T):
            temp_features = []  # Reset features
            for cell_idx in range(len(temp_data_handle[0])):  # Cell number; n
                # Extract features; q
                temp_features.extend(select_feature_from_random_bout(temp_data_handle, cell_idx, q))
            # Merge to output
            temp_data_output.append(temp_features)

        # Store social data to output dictionary with same label keys
        feature_data[key] = np.array(temp_data_output)  # as numpy array

    return feature_data


'''
Group간 비교를 위해 class를 만드는 함수
'''


def make_class(data: dict,
               class_org: list):
    # Return
    data_class = {}
    trial_labels = {}

    for idx, org in enumerate(class_org):
        temp = []
        temp_tiral_labels = []
        for key in org:
            temp.extend(data[key])
            temp_tiral_labels.extend([ord(key)] * len(data[key]))
        data_class[idx] = np.array(temp)
        trial_labels[idx] = temp_tiral_labels

    return data_class, trial_labels


'''
Null model을 만들기 위해 null dataset을 만드는 함수
Null dataset은 label을 randomly shuffle (permutation)하여 패턴을 없애서 만듦
'''


def make_null_dataset(label: np.ndarray):
    np.random.shuffle(label)
    return np.array(label)


'''
Train dataset을 shuffle (permutation)하기 위해 dataset과 label을 함께 shuffle하는 함수
(Data-label pair가 손상되지 않음)
'''


def mix_pair(a: np.ndarray, b: np.ndarray):
    MIX = list(zip(a, b))
    np.random.shuffle(MIX)
    a_mixed, b_mixed = zip(*MIX)
    return np.array(a_mixed), np.array(b_mixed)


'''
Train dataset을 shuffle (permutation)하기 위해 dataset과 trial label, label을 함께 shuffle하는 함수
(Data-trial-label pair가 손상되지 않음)
'''


def mix_3_pair(a: np.ndarray, b: np.ndarray, c: np.ndarray):
    MIX = list(zip(a, b, c))
    np.random.shuffle(MIX)
    a_mixed, b_mixed, c_mixed = zip(*MIX)
    return np.array(a_mixed), np.array(b_mixed), np.array(c_mixed)


'''
Dataset을 만드는 함수
이미 만들어진 data class를 기반으로 concatenate하여 단일 dataset으로 만들고, 그에 맞는 label을 생성함
'''


def make_dataset(data_class: dict,
                 shuffle: bool = True,
                 null_mode: bool = False,
                 **kwargs):
    # NOTICE: This is for binary classes.
    # NULL_MODE: If it was TRUE, Train_dataset and Train_label are randomly shuffled each other.
    # However, Test_dataset and Test_label are not shuffled.

    # kwargs
    trial_labels = []
    if 'trial_labels' in kwargs.keys():
        trial_labels = kwargs.get('trial_labels')
        trial_labels = np.concatenate([trial_labels[0], trial_labels[1]], axis=0)

    # Divide classes
    class_A = data_class[0]  # Class 0
    class_B = data_class[1]  # Class 1

    # Make dataset and labels
    dataset = np.concatenate([class_A, class_B], axis=0)
    label = np.concatenate([np.zeros(len(class_A)), np.ones(len(class_B))], axis=0)

    # Random shuffle for dataset
    if shuffle:
        if len(trial_labels) > 0:
            # Shuffle dataset-trial-label pair
            dataset, trial_labels, label = mix_3_pair(dataset, trial_labels, label)
        else:
            # Shuffle dataset-label pair
            dataset, label = mix_pair(dataset, label)

    # Make null dataset
    if null_mode:
        label = make_null_dataset(label)

    # Return
    if len(trial_labels) > 0:
        return dataset, trial_labels, label
    else:
        return dataset, label


'''
데이터 차원 축소를 위한 reducer를 만드는 함수
'''


def make_reducer(method: str = 'pca', dim: int = 3):
    # Make reducer
    if method == 'pca':
        # Make reducer
        reducer = sklearn.decomposition.PCA(n_components=dim)
    elif method == 'umap':
        # Make reducer
        reducer = umap.UMAP(n_components=dim, metric='cosine', n_neighbors=30, min_dist=0.6)
    else:
        # Make reducer
        reducer = sklearn.decomposition.PCA(n_components=dim)

    return reducer


'''
고차원 데이터를 시각화하기 위해 차원을 축소하는 함수
'''


def reduce_dimension(dataset: dict, reducer: any, fit_transform: bool = False):
    # Return
    emb_dataset = {}

    # Temporary merge dataset
    temp_dataset = []
    temp_length = {}

    for key in dataset.keys():
        temp_length[key] = len(dataset[key])
        temp_dataset.extend(dataset[key])

    # Embedding
    if fit_transform:
        temp_emb_dataset = np.array(reducer.fit_transform(temp_dataset))
    else:
        temp_emb_dataset = np.array(reducer.transform(temp_dataset))

    # Unpack dataset
    for key, data_length in temp_length.items():
        emb_dataset[key] = temp_emb_dataset[0:data_length, :]
        # Remove unpacked data
        temp_emb_dataset = temp_emb_dataset[data_length:]

    return emb_dataset


'''
두 점 사이의 유클리드 거리를 계산하는 함수
'''


def euclidean_distance(a: np.ndarray, b: np.ndarray):
    return np.sqrt(np.sum((a - b) ** 2, axis=0))


'''
축소된 차원 공간에서 클러스터의 중심을 구하는 함수
'''


def cluster_centroid(emb_dataset: np.ndarray):
    # Return
    centroid = []

    # Calculate cluster centroid
    for dim in range(emb_dataset.shape[-1]):
        centroid.append(np.mean(emb_dataset[:, dim], axis=0))

    return centroid


'''
모든 group에 대해서 클러스터의 중심을 구하는 함수
'''


def calc_centroid(dataset: dict):
    # Return
    centroids = {}

    # Calculate cluster centroid
    for key in dataset.keys():
        centroids[key] = cluster_centroid(dataset[key])

    return centroids


'''
축소된 차원 공간에서 클러스터의 반지름을 구하는 함수
'''


def cluster_radius(emb_dataset: np.ndarray):
    # Calculate centroid
    centroid = cluster_centroid(emb_dataset)

    # Calculate radius
    radius = np.sqrt(np.sum((euclidean_distance(emb_dataset, centroid)) ** 2, axis=0) / len(emb_dataset))

    return radius


'''
모든 group에 대해서 클러스터의 반지름을 구하는 함수
'''


def calc_radius(dataset: dict):
    # Return
    radius = {}

    # Calculate cluster radius
    for key in dataset.keys():
        radius[key] = cluster_radius(dataset[key])

    return radius


'''
클러스터 중심 (centroid) 좌표를 입력받아 둘 사이의 유클리드 거리를 계산하는 함수
'''


def centroid_distance(centroid_coord1, centroid_coord2):
    # Check validity
    if len(centroid_coord1) != 3 and len(centroid_coord2) != 3:
        raise ValueError('Check input data: 3D coordinates do not match.')

    # Calculate euclidean distance in 3d plane
    distance = np.sqrt(np.sum(np.square(np.array(centroid_coord1) - np.array(centroid_coord2))))

    return distance


'''
입력받은 모든 클러스터 간 유클리드 거리를 계산하는 함수
'''


def calculate_cluster_centroid_distance(centroid: dict):
    # Return
    distance = {}

    # Get all keys
    keys = list(centroid.keys())

    # Calculate cluster centroid distances for all pairs
    for i, key1 in enumerate(keys[:-1]):
        for key2 in keys[i + 1:]:
            distance[key1 + key2] = centroid_distance(centroid[key1], centroid[key2])

    return distance


'''
입력받은 모든 클러스터 centroid를 각 vertex로 하는 polygon의 면적을 계산하는 함수 (수정중)
'''


def calculate_cluster_polygon_area(centroid: dict):
    # Return
    area = []

    # Check validity
    if len(centroid) < 3:
        raise ValueError('Input centroids do not form a polygon. Number of centroids have to be larger than 3.')

    # Get centroid values
    values = list(centroid.values())

    # Set v0
    v0 = values.pop(0)

    # Calculate polygon area
    for i in range(len(values) - 1):  # Exclude for last pair
        area.append(calculate_vector_triangle_area(v0=v0, v1=values[i], v2=values[i + 1]))
    area = np.sum(area)  # Sum all triangles

    return area


'''
입력받은 세 벡터 (v0, v1, v2)를 vertex로 하는 삼각형의 면적을 계산하는 함수
'''


def calculate_vector_triangle_area(v0: [np.ndarray, list], v1: [np.ndarray, list], v2: [np.ndarray, list]):
    # Check input data
    if len(v0) < 2 or len(v1) < 2 or len(v2) < 2:
        raise ValueError('Each vector have to be larger than 2-dimension.')

    # Calculate triangle area
    u1 = np.array(v1) - np.array(v0)
    u2 = np.array(v2) - np.array(v0)

    area = (1 / 2) * np.sqrt(np.sum(np.cross(u1, u2) ** 2))

    return area


'''
입력받은 org에 대한 클러스터 중심 (centroid)거리를 계산하는 함수
'''


def calculate_org_centroid_distance(centroid: dict, org: list):
    # Return
    distance = 0
    key = ''

    # Get keys in org
    org_keys = np.array(org).reshape(-1)
    key = org_keys[0] + org_keys[1]

    if len(org_keys) == 2:  # Between two clusters
        distance = centroid_distance(centroid[org_keys[0]], centroid[org_keys[1]])

    return key, distance
