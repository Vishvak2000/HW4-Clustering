import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        if not isinstance(k, int):
            raise ValueError("k must be an integer")
        if k < 1:
            raise ValueError("k must be greater than 0")
        
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        self.centroids = None
        

    def kmeans_plus_plus_init(self, mat):
        """
        Initializes centroids using k-means++ method
        """
        n, _ = mat.shape
        centroids = [mat[np.random.choice(n)]]  

        for _ in range(1, self.k):
            dists = np.min(cdist(mat, np.array(centroids)), axis=1) ** 2  # Squared minimum distance (to centroid list), square to give weight to larger distances
            probs = dists / dists.sum()  # Normalize to get probability distribution
            next_centroid = mat[np.random.choice(n, p=probs)]  # Select based on probability
            centroids.append(next_centroid)

        return np.array(centroids)

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features\
        """

        if mat.shape[0] < self.k:
            raise ValueError("The number of data points must be greater than or equal to the number of clusters")
        # Lloyds algorithm
        # 1. Initialize k centroids
        # 2. Assign each data point to the nearest centroid
        # 3. Update the centroids to the mean of the data points assigned to it
        # 4. Repeat steps 2 and 3 until convergence
            # Convergence is defined as the euclidean distance between the old and new centroids being less than a tolerance -> but get the max of that like the slides


        # initialize centroids
        
        self.centroids = self.kmeans_plus_plus_init(mat)
        for _ in range(self.max_iter):
            # assign each data point to the nearest centroid
            dists = cdist(mat, self.centroids)
            labels = np.argmin(dists, axis=1) # get labels based on closest centroid
            # update centroids
            new_centroids = np.array([np.mean(mat[labels == i], axis=0) for i in range(self.k)]) #calculate mean of data points assigned to each centroid
            if max(np.linalg.norm(new_centroids - self.centroids,axis=1)) < self.tol:
                break
            
            self.centroids = new_centroids

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        if mat.shape[1] != self.centroids.shape[1]:
            raise ValueError("The number of features in `mat` must match the number of features in the data the model was fit on")
        return np.argmin(cdist(mat, self.centroids), axis=1)
    
    def get_error(self, mat: np.ndarray) -> float: #passing matrix
        """
        Returns the final squared-mean error of the fit model.  ̶Y̶o̶u̶ ̶c̶a̶n̶ ̶e̶i̶t̶h̶e̶r̶ ̶d̶o̶ ̶t̶h̶i̶s̶ ̶b̶y̶ ̶s̶t̶o̶r̶i̶n̶g̶ ̶
        t̶h̶e̶ ̶o̶r̶i̶g̶i̶n̶a̶l̶ ̶d̶a̶t̶a̶s̶e̶t̶ ̶o̶r̶ ̶r̶e̶c̶o̶r̶d̶i̶n̶g̶ ̶i̶t̶ ̶f̶o̶l̶l̶o̶w̶i̶n̶g̶ ̶t̶h̶e̶ ̶e̶n̶d̶ ̶o̶f̶ ̶m̶o̶d̶e̶l̶ ̶f̶i̶t̶t̶i̶n̶g̶.̶
        I disagree with this - I think we should just pass the matrix itself

        outputs:
            float
                the squared-mean error of the fit model
        """

        return np.mean(np.min(cdist(self.centroids,mat),axis=1) ** 2)

    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centroids