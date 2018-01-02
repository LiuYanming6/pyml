# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:52:23 2017

@author: liuyanming
随机梯度下降 stochastic gradient descent
应用于
    大规模机器学习
    在线学习(实时训练)
优点- 相比批量
    权重更新频繁，更快收敛
    更容易跳出局部最优点，接近全局最有
缺点
    误差曲线不太平滑

np.array 需要注意的一点
b.shape[0] 行数     b.shape[1] 列数
但是当行数为1时，如果没有reshape，b.shape[0]为列数
所以最好都要reshape，避免出现莫名的错误
"""

from numpy.random import seed
import numpy as np

class AdalineSGD(object):
    """Adaptive linear neuron classifier.
    线性分类算法 2优化
    Parameters
    ------------
    eta : float
    n_ter : int

    Attributes
    -----------
    w_ : 1d-arra
    errors_ : list
    shuffle : bool (default: True)
        每次迭代前，打乱
    random_state : int (default: None)
        Set random state for shuffling
        and init the weights

    """
    def __init__(self, eta=0.01, n_iter=10,
                 shuffle=True, random_state=None):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        if random_state:
            seed(random_state)

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)

        return self

    def partial_fit(self, X, y):
        """处理流数据的在线学习"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
#        if y.ravel().shape[0] > 1:
        if y.shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def _shuffle(self, X, y):
        """打乱训练数据集 -- 所以叫随机...
        permutation 排列
        a -> array([[1, 2],
                    [3, 4],
                    [5, 6]])
        len(a) == 3
        b = np.random.permutation(len(a))
        b    -> array([1, 2, 0])
        a[b] -> array([[3, 4],
                      [5, 6],
                      [1, 2]])
        """
        r = np.random.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        self.w_ = np.zeros(1 + m) # -> (1, 3)
        self.w_initialized = True

    def _update_weights(self, xi, target):
        output = self.net_intput(xi)
        error = target - output
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0]  += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def net_intput(self, X):
        """计算 net input"""
        print(self.w_)
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """计算线性激活函数，好像也叫激励函数，反正就是
        linear activation function, 这里是恒等函数
        增加sigmoid函数，好像就是逻辑回归了
        """
        return self.net_intput(X)

    def predict(self, X):
        """Return class label"""
        return np.where(self.activation(X) >= 0.0, 1, -1)

import pandas as pd
df = pd.read_csv('iris.data.txt')
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)
X = df.iloc[0:100, [0, 2]].values

X_std = np.copy(X)
X_std[:,0] = (X_std[:,0] - np.mean(X_std[:,0])) / np.std(X_std[:,0])
X_std[:,1] = (X_std[:,1] - np.mean(X_std[:,1])) / np.std(X_std[:,1])

import matplotlib.pyplot as plt
from plot_decision_regions import plot_decision_regions
ada = AdalineSGD(n_iter=10, eta=0.01).fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
# plt 不支持中文
plt.title('adaline 论标准化的重要性 ')
plt.xlabel('sepal len [std]')
plt.ylabel('petal len [std]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='s')
plt.xlabel('Epochs')
plt.ylabel('平均 cost')
plt.show()










