# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 11:29:25 2017

@author: liuya
"""

import numpy as np
import  matplotlib.pyplot as plt

"""
使用SVM 解决非线性问题
"""
np.random.seed(0)
X_xor = np.random.randn(200,2)
y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)

plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
            c='b', marker='x', label='1')
#plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
#            c='r', marker='s', label='-1')
plt.scatter(X_xor[y_xor==0, 0], X_xor[y_xor==0, 1],
            c='r', marker='s', label='-1')
plt.ylim(-3.0)
plt.legend()
plt.show()

#from sklearn.linear_model import SGDClassifier
"""
RBF kernel又叫高斯核， 核函数
gamma：理解为高斯球面的截至参数(cutoff parameter)
        --> 大 更加紧凑 过拟合
        --> 小 增大受影响的训练样本的范围，导致边界更加宽松
"""
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions
svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
svm.fit(X_xor, y_xor)
plot_decision_regions(X_xor, y_xor, classifier=svm)
plt.legend(loc='upper left')
plt.show()

