import numpy as np
from collections import Counter

class KNNClassifier:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        
        Parameters:
        X (np.array): Training data features.
        y (np.array): Training data labels.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the class labels for the provided data.
        
        Parameters:
        X (np.array): Data to predict.
        
        Returns:
        np.array: Predicted class labels.
        """
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        """
        Predict the class label for a single data point.
        
        Parameters:
        x (np.array): Single data point.
        
        Returns:
        int: Predicted class label.
        """
        # Compute distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

    @staticmethod
    def _euclidean_distance(x1, x2):
        """
        Compute the Euclidean distance between two vectors.
        
        Parameters:
        x1, x2 (np.array): Two vectors to compute the distance between.
        
        Returns:
        float: Euclidean distance.
        """
        return np.sqrt(np.sum((x1 - x2) ** 2))

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load dataset
    iris = load_iris()
    X, y = iris.data, iris.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and fit the model
    knn = KNNClassifier(k=3)
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
