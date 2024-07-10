import numpy as np
from sklearn import datasets

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticRegression(object):
    """
    Logistic regression model.
    """

    def __init__(self, W=None):
        self.weights = W
        self.training_loss = None  # Stores training loss after running gradient descent

    def fit(self, X, y, method='SGD', epochs=None, learning_rate=None, batch_size=None):
        """
        Fit logistic regression model given training data and target array

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features
        y : np.array
            Target values
        method : str
            'SGD' for batch gradient descent
        epochs : int
            Number of epochs to run gradient descent
        learning_rate : float
            Learning rate for gradient descent
        batch_size : int
            Batch size for batch gradient descent

        Returns
        -------
        None

        """
        X = np.insert(X, 0, values=1, axis=1)

        if method == 'SGD':
            self._gradient_descent(X, y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)

        return None

    def predict(self, X):
        X = np.insert(X, 0, values=1, axis=1)
        y_pred = sigmoid(np.dot(X, self.weights))
        return y_pred

    def _gradient_descent(self, X, y, epochs, learning_rate, batch_size):
        num_feats = X.shape[1]
        num_samples = X.shape[0]

        y = y.reshape(num_samples, 1)
        W = np.random.rand(num_feats, 1)
        training_loss_epochs = []

        for ix in range(epochs):
            shuffled_ix = (np.arange(0, len(X)))
            np.random.shuffle(shuffled_ix)
            X = X[shuffled_ix, :]
            y = y[shuffled_ix, :]

            for batch_ix in np.arange(0, X.shape[0], batch_size):
                dW = self._compute_gradient(W, X[batch_ix:batch_ix + batch_size], y[batch_ix:batch_ix + batch_size])
                W -= learning_rate * dW

            if ix % 10 == 0:
                y_pred = sigmoid(np.dot(X, W))
                training_loss = self.logloss(y, y_pred)
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
                training_loss_epochs.append(training_loss)

        self.weights = W
        self.training_loss = training_loss_epochs
        return None

    @staticmethod
    def _compute_gradient(W, X, y):
        y_pred = sigmoid(np.dot(X, W))
        return np.dot(X.T, (y_pred - y)) / len(X)

    @staticmethod
    def logloss(y, y_pred):
        return np.sum((-y*np.log(y_pred) - (1-y)*np.log(1-y_pred)))/len(y)


if __name__ == '__main__':
    
    X, y = datasets.make_classification(n_samples=1000, n_features=5,
                                        n_informative=2, n_redundant=3,
                                        random_state=42)
    # Fit model
    model = LogisticRegression()
    model.fit(X, y, method='SGD', batch_size=250, learning_rate=0.0001, epochs=100)

    # Predict
    y_pred = model.predict(X)
    print(y_pred)