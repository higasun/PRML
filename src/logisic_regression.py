import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models import LogisticRegression
from utils import accuracy
import matplotlib.pyplot as plt
import numpy as np

def plot(cls1, cls2, line):
    x,y = cls1.T
    plt.plot(x, y, 'bo', ms=3, label='class1')
    x,y = cls2.T
    plt.plot(x, y, 'ro', ms=3, label='class2')

    plt.plot(line[0], line[1], color='green', ms=5, label='logistic regression')

    plt.xlim(-10,10)
    plt.ylim(-10,10)

    plt.legend()
    plt.show()

if __name__ == '__main__':
    
    #テスト用2次元データ
    cov = [[4,3],[4,2]] # 共分散
    cls1 = np.random.multivariate_normal([-1.8, 1.8], cov, 100)
    cls2 = np.random.multivariate_normal([1.8, -1.8], cov, 100)

    X = np.concatenate((cls1, cls2), axis=0)
    y = np.array([0]*len(cls1) + [1]*len(cls2))

    # Logistic Regression
    clf = LogisticRegression()
    clf.fit(X, y)
    a = - clf.w[1] / clf.w[2]
    b = - clf.w[0] / clf.w[2]
    x_ = np.linspace(-8, 8, 1000)
    y_ = a * x_ + b
    
    # plot
    plot(cls1, cls2, (x_, y_))

    
    y_pred = clf.predict(X)
    print(accuracy(y_pred, y))
   