import numpy as np


class MyLogisticRegression:
    def __init__(self, learning_rate=0.001, max_iter=10000):
        self._theta = None
        self.intercept_ = None
        self.coef_ = None
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    @staticmethod
    def _sigmoid(z):
        return 1. / (1. + np.exp(-z))

    def fit(self, x_train, y_train):
        def J(theta, X_b, y_train):
            y_hat = self._sigmoid(X_b.dot(theta))
            return - np.sum(y_train * np.log(y_hat) + (1 - y_train) * np.log(1 - y_hat)) / len(y_train)

        def dJ(theta, X_b, y_train):
            y_hat = self._sigmoid(X_b.dot(theta))
            return X_b.T.dot(y_hat - y_train) / len(y_train)

        X_b = np.hstack([np.ones((len(x_train), 1)), x_train])
        self._theta = np.random.randn(X_b.shape[1])
        iter_num = 0
        while iter_num < self.max_iter:
            iter_num += 1
            last_theta = self._theta
            self._theta = self._theta - self.learning_rate * dJ(self._theta, X_b, y_train)
            if abs(J(self._theta, X_b, y_train) - J(last_theta, X_b, y_train)) < 1e-7:
                break

        self.intercept_ = self._theta[0]
        self.coef_ = self._theta[1:]
        return self

    def predict(self, x_predict):
        X_b = np.hstack([np.ones((len(x_predict), 1)), x_predict])
        y_predict = self._sigmoid(X_b.dot(self._theta))
        y_predict = np.array(y_predict >= 0.5, dtype='int')
        return y_predict

    def __repr__(self):
        return "LogisticRegression()"