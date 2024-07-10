import numpy as np

class KMeans:
    def __init__(self, k, num_iters=10):
        self.k = k
        self.num_iters = num_iters
        self.centroids = None

    def fit(self, X):
        """
        Fits the KMeans model to the data X.

        Parameters
        ----------
        X : np.array
            Data in the shape (examples, features).

        Returns
        -------
        clusters : np.array
            The cluster for each data point.
        centroids: np.array
            The centroids of each cluster.
        """
        num_feats = X.shape[1]
        self.centroids = np.random.randn(self.k, num_feats)

        for _ in range(self.num_iters):
            # Compute distance from points to each centroid
            D = self.compute_distances(X)

            # Allocate a point to each cluster
            clusters = D.argmin(axis=1)

            # Compute new centroids
            for i in range(self.k):
                self.centroids[i, :] = X[clusters == i, :].mean(axis=0)

        return clusters, self.centroids

    def compute_distances(self, X):
        """
        Compute euclidean distance between each data point and each centroid.
        """
        num_examples = X.shape[0]
        D = np.zeros((num_examples, self.k))
        for i in range(self.k):
            D[:, i] = self.euclidean_dist(X, self.centroids[i, :])
        return D

    @staticmethod
    def euclidean_dist(x, y):
        return np.sqrt(np.sum((x - y) ** 2, axis=1))


if __name__ == '__main__':
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=2)
    kmeans = KMeans(k=2, num_iters=10)
    clusters, centroids = kmeans.fit(X)

    print("Clusters:\n", clusters)
    print("Centroids:\n", centroids)
