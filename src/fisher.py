import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from models import Fisher, NaiveFisher
import matplotlib.pyplot as plt
import numpy as np

def plot(cls1, cls2, line1, line2):
    x,y = cls1.T
    plt.plot(x, y, 'bo', ms=3, label='class1')
    x,y = cls2.T
    plt.plot(x, y, 'ro', ms=3, label='class2')

    plt.plot(line1[0], line1[1], color='green', ms=5, label='fisher')
    plt.plot(line2[0], line2[1], color='k', ms=5, label='naive')

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

    # Fisher
    clf = Fisher()
    clf.fit(X, y)
    a = - clf.w[0] / clf.w[1]
    b = np.dot(clf.w.T, clf.m) / clf.w[1]
    x1 = np.linspace(-8, 8, 1000)
    y1 = a * x1 + b

    # Naive Fisher
    clf = NaiveFisher()
    clf.fit(X, y)
    a = - clf.w[0] / clf.w[1]
    b = np.dot(clf.w.T, clf.m) / clf.w[1]
    x2 = np.linspace(-8, 8, 1000)
    y2 = a * x2 + b

    # plot
    plot(cls1, cls2, (x1, y1), (x2, y2))
