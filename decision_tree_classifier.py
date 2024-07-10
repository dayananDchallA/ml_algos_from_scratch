import numpy as np

class DecisionTreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
        predicted_class = np.argmax(num_samples_per_class)
        node = DecisionTreeNode(
            gini=self._gini_impurity(y),
            num_samples=len(y),
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )

        if depth < self.max_depth and len(y) >= self.min_samples_split:
            feature_index, threshold, left_indices, right_indices = self._best_split(X, y)
            if left_indices is not None and right_indices is not None:
                node.feature_index = feature_index
                node.threshold = threshold
                node.left = self._grow_tree(X[left_indices, :], y[left_indices], depth + 1)
                node.right = self._grow_tree(X[right_indices, :], y[right_indices], depth + 1)
        return node

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def _predict(self, inputs):
        node = self.root
        while node.left:
            if inputs[node.feature_index] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class

    def _gini_impurity(self, y):
        m = len(y)
        if m == 0:
            return 0
        unique, counts = np.unique(y, return_counts=True)
        probabilities = counts / m
        return 1 - np.sum(probabilities ** 2)

    def _information_gain(self, y, y_left, y_right):
        weight_left = len(y_left) / len(y)
        weight_right = len(y_right) / len(y)
        return self._gini_impurity(y) - (weight_left * self._gini_impurity(y_left) + weight_right * self._gini_impurity(y_right))

    def _best_split(self, X, y):
        best_gain = 0
        best_split_feature = None
        best_split_value = None
        best_left_indices = None
        best_right_indices = None

        for feature_index in range(X.shape[1]):
            values = X[:, feature_index]
            for split_value in np.unique(values):
                left_indices = np.where(values <= split_value)[0]
                right_indices = np.where(values > split_value)[0]

                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue

                y_left = y[left_indices]
                y_right = y[right_indices]
                gain = self._information_gain(y, y_left, y_right)

                if gain > best_gain:
                    best_gain = gain
                    best_split_feature = feature_index
                    best_split_value = split_value
                    best_left_indices = left_indices
                    best_right_indices = right_indices

        return best_split_feature, best_split_value, best_left_indices, best_right_indices

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
    clf = DecisionTreeClassifier(max_depth=3)
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
