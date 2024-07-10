import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class DBSCAN:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps  # Epsilon radius
        self.min_samples = min_samples  # Minimum number of points to form a dense region
        self.labels = None  # Cluster labels

    def fit_predict(self, X):
        """
        Fit the DBSCAN clustering algorithm on the given data and return cluster labels.

        Parameters:
        X (np.array): Input data points.

        Returns:
        np.array: Cluster labels for each data point.
        """
        self.labels = np.zeros(len(X), dtype=int)  # Initialize labels, 0 means unvisited

        cluster_id = 0
        for i in range(len(X)):
            if self.labels[i] != 0:  # Point is already visited
                continue
            
            neighbors = self._find_neighbors(X, i)
            if len(neighbors) < self.min_samples:
                self.labels[i] = -1  # Mark as noise
            else:
                cluster_id += 1
                self._expand_cluster(X, i, neighbors, cluster_id)
        
        return self.labels

    def _find_neighbors(self, X, i):
        """
        Find all points within epsilon radius of point X[i].

        Parameters:
        X (np.array): Input data points.
        i (int): Index of the point to find neighbors for.

        Returns:
        list: Indices of neighbor points.
        """
        neighbors = []
        for j in range(len(X)):
            if np.linalg.norm(X[i] - X[j]) <= self.eps:
                neighbors.append(j)
        return neighbors

    def _expand_cluster(self, X, i, neighbors, cluster_id):
        """
        Expand the cluster by adding reachable points to the current cluster.

        Parameters:
        X (np.array): Input data points.
        i (int): Index of the point to expand cluster around.
        neighbors (list): Indices of neighbor points.
        cluster_id (int): ID of the current cluster.
        """
        self.labels[i] = cluster_id
        j = 0
        while j < len(neighbors):
            neighbor = neighbors[j]
            if self.labels[neighbor] == -1:  # Previously marked as noise
                self.labels[neighbor] = cluster_id
            elif self.labels[neighbor] == 0:  # Unvisited
                self.labels[neighbor] = cluster_id
                new_neighbors = self._find_neighbors(X, neighbor)
                if len(new_neighbors) >= self.min_samples:
                    neighbors += new_neighbors
            j += 1

if __name__ == '__main__':
    # # Generate synthetic data
    # X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.6, random_state=0)

    # # Apply DBSCAN clustering
    # dbscan = DBSCAN(eps=0.5, min_samples=5)
    # labels = dbscan.fit_predict(X)

    # # Plotting the clusters
    # unique_labels = np.unique(labels)
    # colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    # for label, color in zip(unique_labels, colors):
    #     if label == -1:
    #         color = 'k'  # Black color for noise points
        
    #     class_member_mask = (labels == label)
    #     xy = X[class_member_mask]
    #     plt.scatter(xy[:, 0], xy[:, 1], color=color, edgecolor='k', s=50)

    # plt.title('DBSCAN Clustering')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.show()

    from sklearn.datasets import load_iris
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply DBSCAN clustering
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    labels = dbscan.fit_predict(X_scaled)

    # Plotting the clusters
    plt.figure(figsize=(8, 6))

    # Plotting clusters with different colors
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        if label == -1:
            color = 'gray'  # Noise points in gray

        class_member_mask = (labels == label)
        xy = X[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], color=color, edgecolor='k', s=50, label=f'Cluster {label}')

    plt.title('DBSCAN Clustering on Iris Dataset')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.legend()
    plt.show()
