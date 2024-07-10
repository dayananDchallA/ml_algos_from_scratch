import numpy as np

class DecisionTreeNode:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index  # Index of the feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.value = value  # Value if the node is a leaf

class DecisionTreeRegressor:
    def __init__(self, min_samples_split=2, max_depth=2):
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.max_depth = max_depth  # Maximum depth of the tree
        self.root = None  # Root node of the tree

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and depth < self.max_depth:
            best_split = self._get_best_split(X, y, num_features)
            if best_split:
                left_subtree = self._grow_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
                right_subtree = self._grow_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
                return DecisionTreeNode(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree)
        
        leaf_value = self._calculate_leaf_value(y)
        return DecisionTreeNode(value=leaf_value)

    def _get_best_split(self, X, y, num_features):
        best_split = {}
        min_mse = float('inf')

        for feature_index in range(num_features):
            feature_values = X[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            for threshold in possible_thresholds:
                left_indices = np.where(feature_values <= threshold)[0]
                right_indices = np.where(feature_values > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                mse = self._calculate_mse(y, left_indices, right_indices)
                if mse < min_mse:
                    min_mse = mse
                    best_split['feature_index'] = feature_index
                    best_split['threshold'] = threshold
                    best_split['left_indices'] = left_indices
                    best_split['right_indices'] = right_indices
        
        return best_split

    def _calculate_mse(self, y, left_indices, right_indices):
        left_mse = np.mean((y[left_indices] - np.mean(y[left_indices])) ** 2)
        right_mse = np.mean((y[right_indices] - np.mean(y[right_indices])) ** 2)
        return (len(left_indices) * left_mse + len(right_indices) * right_mse) / (len(left_indices) + len(right_indices))

    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _predict(self, inputs):
        node = self.root
        while node.value is None:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Load dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Create and fit the model
    regressor = DecisionTreeRegressor(min_samples_split=10, max_depth=5)
    regressor.fit(X_train, y_train)

    # Make predictions
    y_pred = regressor.predict(X_test)

    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error:", mse)
