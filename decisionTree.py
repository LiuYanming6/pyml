# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 15:51:08 2017

@author: liuya
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn_datasets import getIris, combinedTrainTest
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = getIris()
X_combined, y_combined, testidx_start, testidx_end = combinedTrainTest(
        X_train, y_train, X_test, y_test)

# criterion 不存度衡量标准
tree = DecisionTreeClassifier(criterion='entropy',
                              max_depth=3, random_state=0)
tree.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=tree, 
                      test_idx=range(testidx_start, testidx_end))

plt.xlabel('petal length[cm]')
plt.ylabel('petal width[cm]')
plt.legend(loc='upper left')
plt.show()


"""
将训练后得到的决策树导出为.dot 格式
GraphViz 程序进行可视化处理
"""
from sklearn.tree import export_graphviz
export_graphviz(tree,
                out_file='tree.dot',
                feature_names=['petal length', 'petal width'])

