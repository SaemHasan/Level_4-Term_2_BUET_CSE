import numpy as np
from metrics import accuracy

class LogisticRegression:
    def __init__(self, params):
        """
        figure out necessary params to take as input
        :param params:
        """
        # todo: implement
        self.epochs = params['epochs']
        self.verbose = params['verbose']
        self.learning_rate = params['learning_rate']
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for epoch in range(self.epochs):
            y_pred = self.sigmoid(np.matmul(X, self.weights) + self.bias)
            loss = self.loss(y_true=y, y_pred=y_pred)
            self.gradient_descent(X, y, y_pred)

            if self.verbose:
                print(f'Epoch: {epoch + 1}, Loss: {loss:.3f}, Accuracy: {accuracy(y, self.predict(X)):.3f}')

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        :param X:
        :return:
        """
        # todo: implement
        y_pred = self.sigmoid(np.matmul(X, self.weights) + self.bias)
        if isinstance(y_pred, np.ndarray):
            y_pred = [1 if i > 0.5 else 0 for i in y_pred]
        else:
            y_pred = 1 if y_pred > 0.5 else 0
        return y_pred
    
    def loss(self, y_true, y_pred):
        """
        function for calculating loss
        :param y_true:
        :param y_pred:
        :return:
        """
        y_one_loss = y_true * np.log(y_pred)
        # print(type(y_pred))
        y_zero_loss = (1 - y_true) * np.log(1 - y_pred)

        return -np.mean(y_one_loss + y_zero_loss)
    
    def sigmoid(self, z_array):
        # print(type(z_array))
        if isinstance(z_array, np.ndarray):
            return np.array([self.sigmoid_function(z) for z in z_array])
        else:
            return self.sigmoid_function(z_array)

    def sigmoid_function(self, z):
        """
        function for calculating sigmoid
        :param z:
        :return:
        """
        if z >= 0:
            return 1 / (1 + np.exp(-z))
        else:
            return np.exp(z) / (1 + np.exp(z))
    
    def gradient_descent(self, X, y_true, y_pred):
        """
        function for calculating gradient descent
        :param X:
        :param y_true:
        :param y_pred:
        :return:
        """
        y_dif = y_pred - y_true

        dw = np.matmul(X.T, y_dif) / X.shape[0]
        db = np.mean(y_dif)

        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db



