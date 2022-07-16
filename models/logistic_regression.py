import numpy as np
from scipy import linalg

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

THRESMIN = 1e-10

class LogisticRegression():
    def __init__(self, tol=0.001, max_iter=5, random_seed=0) -> None:
        self.w = None
        self.tol = tol
        self.random_state = np.random.RandomState(random_seed)
        self.max_iter = max_iter
        pass

    def fit(self, X: np.array, t: np.array) -> None:
        self.w  = self.random_state.randn(X.shape[1] + 1)
        Xtil =  np.c_[np.ones(X.shape[0]), X]
        w_prev = self.w
        diff = np.inf
        iteration = 0

        while diff > self.tol and iteration < self.max_iter:
            y = sigmoid(np.dot(Xtil, self.w))
            r = np.clip(y * (1 - y), THRESMIN, np.inf)
            XR = Xtil.T * r
            XRX = np.dot(XR, Xtil)
            w_prev = self.w
            b = np.dot(XR, np.dot(Xtil, self.w) - 1/r * (y - t))
            self.w = linalg.solve(XRX, b)
            diff = abs(self.w - w_prev).mean()
            iteration += 1

    def predict(self, X):
        Xtil = np.c_[np.ones(X.shape[0]), X]
        y = sigmoid(np.dot(Xtil, self.w))
        return np.where(y > 0.5, 1, 0)