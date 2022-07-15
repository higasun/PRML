import numpy as np

class NaiveFisher():
    def __init__(self):
        self.w = None
        self.m = None
        self.class_num = None
        return

    def fit(self, X, y):
        # Number of classes
        self.dim = X.shape[1]

        X1 = X[y == 0]
        X2 = X[y == 1]

        m1 = np.mean(X1, axis=0)
        m2 = np.mean(X2, axis=0)

        self.w = (m2 - m1)

        self.m = np.mean(X, axis=0)

    def predict(self, X):
        y = np.dot(self.w.T, (X - self.m))
        y = y[y < 0].astype(np.int)
        return y


