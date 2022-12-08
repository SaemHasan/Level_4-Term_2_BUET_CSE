from data_handler import bagging_sampler
import numpy as np

class BaggingClassifier:
    def __init__(self, base_estimator, n_estimator):
        """
        :param base_estimator:
        :param n_estimator:
        :return:
        """
        # todo: implement
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self.estimators = []

    def fit(self, X, y):
        """
        :param X:
        :param y:
        :return: self
        """
        assert X.shape[0] == y.shape[0]
        assert len(X.shape) == 2
        # todo: implement
        for i in range(self.n_estimator):
            X_sample, y_sample = bagging_sampler(X, y)
            estimator = self.base_estimator
            estimator.fit(X_sample, y_sample)
            self.estimators.append(estimator)

    def predict(self, X):
        """
        function for predicting labels of for all datapoint in X
        apply majority voting
        :param X:
        :return:
        """
        # todo: implement
        y_pred = []
        for i in range(X.shape[0]):
            y_pred.append(self.majority_voting(X[i]))
        y_pred = np.array(y_pred)
        return y_pred
    
    def majority_voting(self, x):
        """
        function for calculating majority voting
        :param x:
        :return:
        """
        y_predictions = []
        for estimator in self.estimators:
            y_pred = estimator.predict(x)
            y_predictions.append(y_pred)
        y_predictions = np.array(y_predictions)
        # print(y_predictions)
        # print("prediction : ",np.argmax(np.bincount(y_predictions)))
        return np.argmax(np.bincount(y_predictions))


