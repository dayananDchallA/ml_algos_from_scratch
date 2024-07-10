import numpy as np
import matplotlib.pyplot as plt

class LinearRegression(object):
    """
    Linear regression model. Can be fit by either OLS or gradient descent.
    """

    def __init__(self):
        self.weights = None
        self.training_loss = None  # Stores training loss after running gradient descent

    def predict(self, X):
        """
        Predict outputs from an array of inputs

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features

        Returns
        -------
        np.array
            Array of predictions from trained linear regression model

        """
        X = np.insert(X, 0, values=1, axis=1)  # Add constant 1 to start of array

        assert X.shape[1] == len(self.weights), "Input size of {0} does not match expected size of {1}".format(
            X.shape[1], len(self.weights))
        return np.dot(X, self.weights)

    def fit(self, X, y, method='OLS', epochs=None, learning_rate=None, batch_size=None):
        """
        Fit linear regression model given training data and target array

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features
        y : np.array
            Target values
        method : str
            'OLS' for ordinary least squares solution, 'SGD' for batch gradient descent
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

        # Add a column of ones to training data
        X = np.insert(X, 0, values=1, axis=1)

        if method == 'OLS':
            ols_weights = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y))
            self.weights = ols_weights
        elif method == 'SGD':
            self._gradient_descent(X, y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size)
        else:
            raise ValueError('Method "{0}" was not recognised. Method must be either "OLS" or "SGD".'.format(method))

        return None

    def _gradient_descent(self, X, y, epochs, learning_rate, batch_size):
        """
        Optimises weights using batch gradient descent

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features
        y : np.array
            Target values
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
                y_pred = np.dot(X, W)
                training_loss = self.mse(y, y_pred)
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
                training_loss_epochs.append(training_loss[0])

        self.weights = W
        self.training_loss = training_loss_epochs
        return None

    @staticmethod
    def _compute_gradient(W, X, y):
        """ Compute gradient

        Returns
        -------
        dW : np.array
            Gradient of weights
        """
        y_pred = np.dot(X, W)
        err = (y - y_pred)
        dW = np.dot(-X.T, err) / len(X)
        return dW

    @staticmethod
    def mse(y, y_pred):
        assert len(y) == len(y_pred)
        loss = (0.5 / len(y)) * np.dot((y - y_pred).T, (y - y_pred))
        return loss

if __name__ == '__main__':
    

    # Generate random data
    n_feats = 1
    n_obs = 10
    W = np.random.random((n_feats, 1))
    X = np.random.randint(0, 100, size=[n_obs, n_feats])
    y = np.dot(X, W) + np.random.rand(n_obs, 1) * 10

    # Fit model
    model = LinearRegression()
    model.fit(X, y, method='SGD', batch_size=250, learning_rate=0.0001, epochs=100)

    # Predict
    y_pred = model.predict(X)

    print(y_pred)

    # # Plot results
    # ax, fig = plt.subplots(figsize=[10, 7.5])
    # plt.scatter(X, y)
    # plt.plot(X, y_pred)
    # plt.show()