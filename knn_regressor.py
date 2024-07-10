import numpy as np

class KNNRegressor:
    def __init__(self, k=5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Store the training data.
        
        Parameters:
        X (np.array): Training data features.
        y (np.array): Training data target values.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Predict the target values for the provided data.
        
        Parameters:
        X (np.array): Data to predict.
        
        Returns:
        np.array: Predicted target values.
        """
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        """
        Predict the target value for a single data point.
        
        Parameters:
        x (np.array): Single data point.
        
        Returns:
        float: Predicted target value.
        """
        # Compute distances between x and all examples in the training set
        distances = [self._euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        
        # Extract the target values of the k nearest neighbor training samples
        k_nearest_values = [self.y_train[i] for i in k_indices]
        
        # Return the mean of the k nearest neighbor values
        return np.mean(k_nearest_values)

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

if __name__ == "__main__":
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    # Load the California Housing dataset
    housing = fetch_california_housing()

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.2, random_state=42)

    # Scale the features for better performance of k-NN
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Create and fit the k-NN regressor
    from sklearn.neighbors import KNeighborsRegressor

    knn_regressor = KNeighborsRegressor(n_neighbors=5)
    knn_regressor.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = knn_regressor.predict(X_test_scaled)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
