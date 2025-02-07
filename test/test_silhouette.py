import numpy as np
from cluster import (
        KMeans, 
        Silhouette, 
        make_clusters,
        plot_clusters,
        plot_multipanel)

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans as KMeans_sklearncore
from sklearn.metrics import silhouette_score

def test_silhouette():
    simple_cluster, labels = make_clusters(scale=0.3)
    kmeans = KMeans(3)
    kmeans.fit(simple_cluster)
    y = kmeans.predict(simple_cluster)
    sil = Silhouette()
    sil_score = sil.score(simple_cluster, y)
    assert np.isclose(sil_score.mean(), silhouette_score(simple_cluster, y).mean(),rtol=1e-2)
