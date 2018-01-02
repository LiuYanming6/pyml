# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 13:38:23 2017

@author: liuyanming
"""

# scikit-learn

from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt

# 1 提取数据
iris = datasets.load_iris()
X = iris.data[:, [2,3]]
y = iris.target

# 2 随机将数据矩阵按 7比3 划分为训练数据集和测试数据集
#print(np.unique(y))
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0)
#print(X_train.shape[0], X_test.shape[0]) # 105 / 45 7 / 3

# 3 标准化处理
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # 计算训练数据中的每个特征的样本均值和便准差
X_train_std = sc.transform(X_train)
X_test_std =  sc.transform(X_test) # 使用相同的缩放参数，保证和训练集彼此相当
#print(X_test)
#print(X_test_std)

# 4 训练感知机
from sklearn.linear_model import Perceptron
# random_state 默认为None, 这里我们让每次迭代后初始化重排训练数据集
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train_std, y_train)

# 5 进行预测
# n_iter 错误
# 40     4
# 60     1
# 100    10
y_pred = ppn.predict(X_test_std)
print('错误预测数:%d' % (y_test != y_pred).sum())
from sklearn.metrics import accuracy_score
print('准确率: %.2f' % accuracy_score(y_test, y_pred))

# 6 决策区域
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))
from plot_decision_regions import plot_decision_regions
plot_decision_regions(X=X_combined_std, y=y_combined,
                      classifier=ppn, test_idx=range(105, 150))
plt.xlabel('petal len(std) ')
plt.ylabel('petal width(std)')
plt.legend(loc='upper left')
plt.show()

# 结论：
#   从图像中我们看出，无法通过一个线性决策边界完美区分三类样本


