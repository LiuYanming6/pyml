# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 13:45:30 2017

@author: liuyanming
"""
import numpy as np

class AdalineGD(object):
    """ADAptive Linear Neuron classifier
    线性分类算法 2
    自适应线性神经元分类器
    参数
    -----------
    eta: float
        学习速率 0.0~1.0
    n_iter: int
        迭代次数

    属性
    -----------
    w_: 1d-array 1 * (n_features + 1)
        训练权重
    errors_: list
        每次编列的错误数
    cost_: list
        存储代价函数的输出值以检验本轮训练后是否收敛
    """
    def __init__(self, eta=0.01, n_iter=50):
        self.eta = eta
        self.n_ter = n_iter

    def fit(self, X, y):
        """训练样本集
        参数
        ----------
        X: array-like, shape = [n_samples, n_features]
            样本向量
        y: array-like, shape = [n_samples]
            标记 target values

        Returns
        ----------
        self: object
        """
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_ter):
            # 不是一次一个样本，所以也叫批量梯度下降
            output = self.net_intput(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_intput(self, X):
        """计算 net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """计算线性激活函数，好像也叫激励函数，反正就是
        linear activation function, 这里是恒等函数
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

# 学习速率的重要性
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(8, 4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X, y)
ax[0].plot(range(1, len(ada1.cost_) + 1),
           np.log10(ada1.cost_), marker='o')
#          这个log好像只是让线条更平滑了，好看的些
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(sum-squared-error)')
ax[0].set_title('adaline - learning rate 0.01')

ada2 = AdalineGD(n_iter=10, eta=0.0001).fit(X, y)
ax[1].plot(range(1, len(ada2.cost_) + 1),
           np.log10(ada2.cost_), marker='s')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(sum-squared-error)')
ax[1].set_title('adaline - learning rate 0.0001')

plt.show()

# 优化算法的性能
# 标准化的特征缩放方法，使数据具备标准正太分布的特性
# 各特征值均值为0 标准差为1 公式 xi' = (xi - 均值) / 标准差
# 同样用学习速率0.01，标准化后就可以收敛了

# 标准化 -- 可以增大学习速率，更快收敛
X_std = np.copy(X)
X_std[:,0] = (X_std[:,0] - np.mean(X_std[:,0])) / np.std(X_std[:,0])
X_std[:,1] = (X_std[:,1] - np.mean(X_std[:,1])) / np.std(X_std[:,1])

from plot_decision_regions import plot_decision_regions
ada = AdalineGD(n_iter=10, eta=0.01).fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
# plt 不支持中文
plt.title('adaline 论标准化的重要性 ')
plt.xlabel('sepal len [std]')
plt.ylabel('petal len [std]')
plt.legend(loc='upper left')
plt.show()

plt.plot(range(1, len(ada.cost_) + 1), ada.cost_, marker='s')
plt.xlabel('Epochs')
plt.ylabel('sum-squared-error')
plt.show()
