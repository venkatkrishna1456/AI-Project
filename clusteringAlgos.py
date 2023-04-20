
import numpy as np
from sklearn.cluster import DBSCAN, AffinityPropagation, KMeans, AgglomerativeClustering

def dbscanClustering(data, eps = 0.5, dist_type = 'euclidean'):
    # loading DBSCAN model
    dbscan = DBSCAN(eps = eps, metric = dist_type)

    # fitting DBSCAN
    dbscan.fit(data)

    return dbscan.labels_

def affinityPropagation(data, dist_type = 'euclidean'):
    # loading AffinityPropagation model
    model = AffinityPropagation(affinity = dist_type)

    # fitting AffinityPropagation
    model.fit(data)

    return model.labels_

def kMeans(data, num_clusters):
    # loading k means clustering model
    kmeans = KMeans(n_clusters=num_clusters)

    #fitting k means clustering
    kmeans.fit(data)
    
    return(kmeans.labels_)

def agglomerativeClustering(data, num_clusters, dist_type = 'euclidean'):
    # loading agglomerative clustering model
    cluster = AgglomerativeClustering(n_clusters = num_clusters, affinity = dist_type, linkage = 'ward')

    # fitting agglomerative clutering model
    cluster.fit_predict(data)

    return cluster.labels_
