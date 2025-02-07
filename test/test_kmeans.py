import pytest 
import numpy as np
from cluster import (
        KMeans, 
        #Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as KMeans_sklearn



def test_kmeans_init():

    with pytest.raises(ValueError):
        kmeans = KMeans(0) # k must be greater than 0
        kmeans = KMeans(1.5) # k must be an integer
        kmeans = KMeans("1") #
    

def test_kmeans_fit():
    #We are also testing predict here
    simple_cluster, labels = make_clusters(scale=0.3)
    
    kmeans = KMeans(3)
    kmeans_sk = KMeans_sklearn(3, init="k-means++") #Import sklearns version
    
    kmeans.fit(simple_cluster)
    kmeans_sk.fit(simple_cluster)


    _, my_kmeans = np.unique(kmeans.predict(simple_cluster), return_counts=True)
    _, sklearn_kmeans = np.unique(kmeans_sk.labels_, return_counts=True)
    assert np.array_equal(np.sort(my_kmeans), np.sort(sklearn_kmeans))
    # assert that cluster assignments are the same


    # assert that K < n
    with pytest.raises(ValueError):
        kmeans = KMeans(1000)
        kmeans.fit(simple_cluster)








