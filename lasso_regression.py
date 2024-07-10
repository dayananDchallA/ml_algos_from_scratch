import numpy as np

class LassoRegression(object):
    """
    Lasso regression model. Can be fit by either OLS or gradient descent.
    """

    def __init__(self, reg_lambda=1.0):
        self.weights = None
        self.training_loss = None  # Stores training loss after running gradient descent
        self.reg_lambda = reg_lambda

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

    def fit(self, X, y, method='CD', epochs=None, learning_rate=None, batch_size=None):
        """
        Fit lasso regression model given training data and target array

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features
        y : np.array
            Target values
        method : str
            'CD' for coordinate descent solution, 'SGD' for batch gradient descent
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

        if method == 'CD':
            self.weights = self._coordinate_descent(X, y, epochs, self.reg_lambda)
        elif method == 'SGD':
            self._gradient_descent(X, y, epochs=epochs, learning_rate=learning_rate, batch_size=batch_size, reg_lambda=self.reg_lambda)
        else:
            raise ValueError('Method "{0}" was not recognised. Method must be either "CD" or "SGD".'.format(method))

        return None

    def _coordinate_descent(self, X, y, epochs, reg_lambda):
        """
        Optimizes weights using coordinate descent

        Parameters
        ----------
        X : np.array
            Input data where rows are observations and columns are features
        y : np.array
            Target values
        epochs : int
            Number of epochs to run coordinate descent
        reg_lambda : float
            L1 regularisation term. The larger `reg_lambda` is the greater the regularisation effect.

        Returns
        -------
        np.array
            Optimized weights

        """
        m, n = X.shape
        weights = np.zeros(n)
        y = y.flatten()

        for epoch in range(epochs):
            for j in range(n):
                residual = y - np.dot(X, weights)
                rho = np.dot(X[:, j], residual + weights[j] * X[:, j])
                
                if j == 0:  # Do not regularize the bias term
                    weights[j] = rho / np.dot(X[:, j], X[:, j])
                else:
                    weights[j] = self.soft_thresholding(rho, reg_lambda) / np.dot(X[:, j], X[:, j])

            if epoch % 10 == 0:
                y_pred = np.dot(X, weights)
                training_loss = self.mse_with_l1(y, y_pred, weights, reg_lambda)
                print(f'epoch {epoch} : training loss {training_loss}')

        return weights

    @staticmethod
    def soft_thresholding(rho, reg_lambda):
        if rho < -reg_lambda:
            return rho + reg_lambda
        elif rho > reg_lambda:
            return rho - reg_lambda
        else:
            return 0

    def _gradient_descent(self, X, y, epochs, learning_rate, batch_size, reg_lambda):
        """
        Optimizes weights using batch gradient descent

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
        reg_lambda : float
            L1 regularisation term. The larger `reg_lambda` is the greater the regularisation effect.

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
                dW = self._compute_gradient(W, X[batch_ix:batch_ix + batch_size], y[batch_ix:batch_ix + batch_size], reg_lambda)
                W -= learning_rate * dW

            if ix % 10 == 0:
                y_pred = np.dot(X, W)
                training_loss = self.mse_with_l1(y, y_pred, W, reg_lambda)
                print('epoch {0} : training loss {1}'.format(ix, training_loss))
                training_loss_epochs.append(training_loss[0])

        self.weights = W
        self.training_loss = training_loss_epochs
        return None

    @staticmethod
    def _compute_gradient(W, X, y, reg_lambda):
        """ Compute gradient of regularised mse loss function

        Returns
        -------
        dW : np.array
            Gradient of weights
        """
        y_pred = np.dot(X, W)
        err = (y - y_pred)
        dW = (np.dot(-X.T, err) / len(X)) + reg_lambda * np.sign(W)
        return dW

    @staticmethod
    def mse_with_l1(y, y_pred, W, reg_lambda):
        assert len(y) == len(y_pred)
        loss = (0.5 / len(y)) * np.dot((y - y_pred).T, (y - y_pred)) + (0.5 * reg_lambda * np.sum(np.abs(W)))
        return loss

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import random
    random.seed(42)

    # Generate random data
    n_feats = 1
    n_obs = 10
    W = np.random.random((n_feats, 1))
    X = np.random.randint(0, 100, size=[n_obs, n_feats])
    y = np.dot(X, W) + np.random.rand(n_obs, 1) * 10
    # Create outlier
    X[-1] = 100
    y[-1] = 300

    # Fit model
    model = LassoRegression(reg_lambda=1.0)
    model.fit(X, y, method='SGD', batch_size=5, learning_rate=0.001, epochs=100)

    # Predict
    y_pred = model.predict(X)
    print(y_pred)
