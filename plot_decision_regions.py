# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 15:15:56 2017

@author: liuyanming
"""

# 4. 实现对二维数据集决策边界的可视化

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=.02):
    # 模型的决策区域
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
#    print(xx1, xx2, z, cmap)
    # 直线分割的两块画布 此处的alpha要小于下面的，不然下面的点画布上
    # FIXME: contourf 需要再理解实践
    plt.contourf(xx1, xx2, z, alpha=0.4, cmap=cmap)

    # 把不同的分类点撒进去
    for idx, c1 in enumerate(np.unique(y)):
        plt.scatter(x=X[y == c1, 0], y=X[y == c1, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=c1)

    # 使用小圆圈高亮显示来自测试数据集的样本
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1],
                    alpha=1.0, linewidths=1, marker='o',
                    s=55, label='test set')
