import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        n = X.shape[0]
        silhouette_scores = np.zeros(n)
        for i in range(n):
            a = np.mean(cdist(X[y == y[i]], [X[i]])) # get mean pariwse distances between point X[i] and all other points with the same label
            b = np.min([np.mean(cdist(X[y == j], [X[i]])) for j in np.unique(y) if j != y[i]]) 
            # for every (unique), label that isnt the same label as the point, get the mean distance of that point and all other points with NOT that label
            silhouette_scores[i] = (b - a) / max(a, b) # just the formula
        return silhouette_scores
