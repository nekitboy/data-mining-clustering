from sklearn.cluster import KMeans
import numpy as np


class KMeansModel(object):

    def __init__(self, n_clusters=10, tol=0.001, n_init=20):
        self.k_means = KMeans(n_clusters, tol=tol, n_init=n_init, max_iter=1000)
        self.labels = None
        self.cluster_centers = None

    def fit(self, X):
        self.k_means.fit(list(X))
        self.labels = self.k_means.labels_
        self.cluster_centers =self.k_means.cluster_centers_