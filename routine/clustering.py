from sklearn.cluster import KMeans


def Kmeans_cluster(feats, n_cluster):
    return KMeans(n_clusters=n_cluster, random_state=1).fit_predict(feats)
