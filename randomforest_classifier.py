import numpy as np
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2):
        self.n_estimators = n_estimators  # Number of trees in the forest
        self.max_depth = max_depth        # Maximum depth of each tree
        self.min_samples_split = min_samples_split  # Minimum samples required to split a node
        self.trees = []  # List to hold the decision trees

    def fit(self, X, y):
        """
        Build a forest of decision trees from the training set (X, y).

        Parameters:
        X (np.array): Training data, shape (n_samples, n_features).
        y (np.array): Target values, shape (n_samples,).
        """
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_estimators):
            # Randomly select samples with replacement (bootstrap)
            bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            # Build a decision tree with random subset of features
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (np.array): Samples, shape (n_samples, n_features).

        Returns:
        np.array: Predicted class labels, shape (n_samples,).
        """
        # Make predictions with each tree
        tree_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Aggregate predictions using majority voting
        y_pred = []
        for i in range(tree_predictions.shape[1]):
            counts = Counter(tree_predictions[:, i])
            y_pred.append(counts.most_common(1)[0][0])

        return np.array(y_pred)

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        """
        Build a decision tree from the training set (X, y).

        Parameters:
        X (np.array): Training data, shape (n_samples, n_features).
        y (np.array): Target values, shape (n_samples,).
        """
        self.tree = self._grow_tree(X, y, depth=0)

    def _grow_tree(self, X, y, depth):
        """
        Recursively grow the decision tree.

        Parameters:
        X (np.array): Training data, shape (n_samples, n_features).
        y (np.array): Target values, shape (n_samples,).
        depth (int): Current depth of the tree.

        Returns:
        dict: Tree node containing split criterion and child nodes.
        """
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # Stopping criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (len(np.unique(y)) == 1) or \
           (n_samples < self.min_samples_split):
            return {'value': Counter(y).most_common(1)[0][0]}

        # Find the best split
        best_split = self._find_best_split(X, y)
        if best_split['impurity'] == 0:
            return {'value': Counter(y).most_common(1)[0][0]}

        # Recursively grow the tree
        left_subtree = self._grow_tree(*best_split['left'], depth + 1)
        right_subtree = self._grow_tree(*best_split['right'], depth + 1)

        return {'feature': best_split['feature'],
                'threshold': best_split['threshold'],
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        """
        Find the best split for a node.

        Parameters:
        X (np.array): Training data, shape (n_samples, n_features).
        y (np.array): Target values, shape (n_samples,).

        Returns:
        dict: Best split criterion (feature index, threshold, impurity).
        """
        n_samples, n_features = X.shape
        if n_samples <= 1:
            return {'impurity': 0}

        # Calculate impurity before split
        parent_impurity = self._calculate_impurity(y)

        best_split = {'feature': None, 'threshold': None, 'impurity': 1}

        # Iterate over all features
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # Try all possible splits
            for threshold in unique_values:
                y_left = y[feature_values <= threshold]
                y_right = y[feature_values > threshold]

                if len(y_left) > 0 and len(y_right) > 0:
                    left_impurity = self._calculate_impurity(y_left)
                    right_impurity = self._calculate_impurity(y_right)

                    impurity = (len(y_left) / n_samples) * left_impurity + \
                               (len(y_right) / n_samples) * right_impurity

                    if impurity < best_split['impurity']:
                        best_split = {'feature': feature_idx,
                                      'threshold': threshold,
                                      'impurity': impurity,
                                      'left': (X[feature_values <= threshold], y[feature_values <= threshold]),
                                      'right': (X[feature_values > threshold], y[feature_values > threshold])}

        return best_split

    def _calculate_impurity(self, y):
        """
        Calculate Gini impurity for a node.

        Parameters:
        y (np.array): Target values, shape (n_samples,).

        Returns:
        float: Gini impurity.
        """
        n_samples = len(y)
        if n_samples == 0:
            return 0

        class_counts = Counter(y)
        impurity = 1 - sum((count / n_samples) ** 2 for count in class_counts.values())

        return impurity

    def predict(self, X):
        """
        Predict class labels for samples in X.

        Parameters:
        X (np.array): Samples, shape (n_samples, n_features).

        Returns:
        np.array: Predicted class labels, shape (n_samples,).
        """
        return np.array([self._predict_tree(x, self.tree) for x in X])

    def _predict_tree(self, x, tree):
        """
        Recursively traverse the decision tree to predict the class label for a sample.

        Parameters:
        x (np.array): Sample, shape (n_features,).
        tree (dict): Decision tree node.

        Returns:
        int: Predicted class label.
        """
        if 'value' in tree:
            return tree['value']

        feature_value = x[tree['feature']]
        if feature_value <= tree['threshold']:
            return self._predict_tree(x, tree['left'])
        else:
            return self._predict_tree(x, tree['right'])

# Example usage:
if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest classifier
    random_forest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2)
    random_forest.fit(X_train, y_train)

    # Make predictions
    y_pred = random_forest.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')
