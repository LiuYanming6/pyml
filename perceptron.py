# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 21:07:53 2017

@author: liu
"""

import numpy as np

class Perceptron(object):
    """线性分类算法 1
    感知器规则算法
    使用单步跳跃函数
    Parameter
    ------------
    eta: float
        description
    n_iter: int

    Attributes
    -----------
    w_: 1d-array
        需要训练的权重
    errors_: list
        每次遍历的错误数
    """

    def __init__(self, ela=0.01, n_iter=10):
        self.eta = ela
        self.n_iter = n_iter


    def fit(self, X, y):
        # 多加了一列
        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            # 因为下面的for循环中有权重更新，不能向量化
            # 每次一个样本渐进更新权重
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)


        print(self.w_)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)


import pandas as pd
#df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = pd.read_csv('iris.data.txt')
#print(df.tail())

import matplotlib.pyplot as plt

# 取前100行的第4列
# iloc interger-location based indexing for selection by position
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# 取前100行的第0,2列
X = df.iloc[0:100, [0, 2]].values

# scatter 散点图
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal len')
plt.ylabel('sepal len')
plt.legend(loc='upper left')  # 图标上的文字说明
plt.show()


# 3. 开始训练，得到权重 w_=[-0.4  -0.68  1.82]
ppn = Perceptron(ela=0.1, n_iter=10)
ppn.fit(X, y)
# plot 折线图
plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_)
plt.xlabel('Epochs 次数')
plt.ylabel('错误数目')
plt.show()

# 4. 实现对二维数据集决策边界的可视化
from matplotlib.colors import ListedColormap
def plot_decision_regions(X, y, classifier, resolution=.02):
    #
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # 为什么 -1 +1 呢
    print(X[:, 0].max())
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    z = z.reshape(xx1.shape)
    print(xx1, xx2, z, cmap)
    # 直线分割的两块画布 此处的alpha要小于下面的，不然下面的点画布上
    # FIXME: contourf 需要再理解实践
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)

    # 把不同的分类点撒进去
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)


plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('petal len')
plt.ylabel('sepal len')
plt.legend(loc='upper left')
plt.show()



