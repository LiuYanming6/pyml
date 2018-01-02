# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:15:31 2017

@author: liuya
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn_datasets import getIris, combinedTrainTest
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = getIris()
X_combined, y_combined, testidx_start, testidx_end = combinedTrainTest(
        X_train, y_train, X_test, y_test)

# criterion 不存度衡量标准
forest = RandomForestClassifier(criterion='entropy',
                              n_estimators=10,      # 10 棵决策树
                              random_state=1,
                              n_jobs=20)         # 所需处理器内核的数量
forest.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=forest, 
                      test_idx=range(testidx_start, testidx_end))

plt.xlabel('petal length[cm]')
plt.ylabel('petal width[cm]')
plt.legend(loc='upper left')
plt.show()
