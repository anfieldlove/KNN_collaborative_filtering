import numpy as np 


def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    dists = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            dists[i, j] = np.sqrt(np.sum(np.square(X[i] - Y[j])))

    return dists


def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    dists = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            dists[i, j] = np.sum(np.abs(X[i] - Y[j]))

    return dists


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """
    dists = np.zeros((len(X), len(Y)))
    for i in range(len(X)):
        for j in range(len(Y)):
            dists[i, j] = 1 - np.dot(X[i], Y[j]) / ((np.linalg.norm(X[i])) * (np.linalg.norm(Y[j])))

    return dists
