import numpy as np
from .k_nearest_neighbor import KNearestNeighbor


def collaborative_filtering(input_array, n_neighbors,
                            distance_measure='euclidean', aggregator='mode'):
    """

    Arguments:
        input_array {np.ndarray} -- An input array of shape (n_samples, n_features).
            Any zeros will get imputed.
        n_neighbors {int} -- Number of neighbors to use for prediction.
        distance_measure {str} -- Which distance measure to use. Can be one of
            'euclidean', 'manhattan', or 'cosine'. This is the distance measure
            that will be used to compare features to produce labels.
        aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
            neighbors. Can be one of 'mode', 'mean', or 'median'.

    Returns:
        imputed_array {np.ndarray} -- An array of shape (n_samples, n_features) with imputed
            values for any zeros in the original input_array.
    """

    for j in range(len(input_array[0])):
        index = []
        targets = np.zeros([len(input_array), 1])
        features = []
        input = input_array
        for i in range(len(input_array)):
            if input_array[i, j] == 0:
                index.append(i)
        if len(index):
            targets = np.column_stack((targets, input_array[:, j]))

            input = np.delete(input_array, j, axis=1)
            features = input[index]
            input = np.delete(input, index, axis=0)
            targets = np.delete(targets, index, axis=0)
            targets = np.delete(targets, 0, axis=1)

            knn = KNearestNeighbor(n_neighbors)
            knn.fit(input, targets)

            labels = knn.predict(features)
            for d in range(len(labels)):
                input_array[index[d], j] = labels[d, 0]

    return input_array





