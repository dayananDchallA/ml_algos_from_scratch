import numpy as np

class TreeNode:
    def __init__(self, feature=None, threshold=None, value=None, left=None, right=None):
        self.feature = feature      # Index of feature to split on
        self.threshold = threshold  # Threshold value for the split
        self.value = value          # Value if the node is a leaf in regression tasks
        self.left = left            # Left child node
        self.right = right          # Right child node

class DecisionTreeRegressor:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth

    def fit(self, X, y):
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        # Initialize with the mean of the targets at this node
        prediction = np.mean(y)
        if depth < self.max_depth:
            feature_indices = np.random.choice(self.n_features_, self.n_features_, replace=True)
            best_gain = 0
            for feature_index in feature_indices:
                thresholds = np.unique(X[:, feature_index])
                for threshold in thresholds:
                    gain = self._information_gain(X, y, feature_index, threshold)
                    if gain > best_gain:
                        best_gain = gain
                        best_feature, best_threshold = feature_index, threshold
            if best_gain > 0:
                left = X[:, best_feature] < best_threshold
                right = X[:, best_feature] >= best_threshold
                return TreeNode(
                    feature=best_feature,
                    threshold=best_threshold,
                    left=self._grow_tree(X[left], y[left], depth + 1),
                    right=self._grow_tree(X[right], y[right], depth + 1))
        return TreeNode(value=prediction)

    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _information_gain(self, X, y, feature_index, threshold):
        left_indices = X[:, feature_index] < threshold
        left_y, right_y = y[left_indices], y[~left_indices]
        mse = 0
        for partition in (left_y, right_y):
            count = len(partition)
            if count == 0:
                continue
            probability = len(partition) / count
            mse += probability * np.var(partition)
        return mse


class RandomForestRegressor:
    def __init__(self, n_estimators=100, max_depth=None, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.estimators = [DecisionTreeRegressor(max_depth=self.max_depth) for _ in range(self.n_estimators)]

    def fit(self, X, y):
        for estimator in self.estimators:
            indices = np.random.choice(len(X), len(X), replace=True)
            X_subset, y_subset = X[indices], y[indices]
            estimator.fit(X_subset, y_subset)

    def predict(self, X):
        return np.mean([estimator.predict(X) for estimator in self.estimators], axis=0)

if __name__ == '__main__':
    from sklearn.datasets import fetch_california_housing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    import numpy as np

    # Load the California housing dataset
    california = fetch_california_housing()
    X = california.data
    y = california.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a RandomForestRegressor
    rf_regressor = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    # Fit the model
    rf_regressor.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf_regressor.predict(X_test)

    # Calculate Mean Squared Error (MSE) as a performance metric
    mse = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error: {mse}")

    # Print a few predictions and actual values
    print("\nPredicted\tActual")
    for i in range(5):
        print(f"{y_pred[i]:.2f}\t\t{y_test[i]:.2f}")

    # Feature importance
    feature_importances = rf_regressor.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    print("\nFeature Importances:")
    for i in sorted_indices:
        print(f"{california.feature_names[i]}: {feature_importances[i]}")
