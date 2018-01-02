# -*- coding: utf-8 -*-
"""
Created on Fri Dec 15 17:24:51 2017

@author: liuya
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn_datasets import getStdXyTrainAndTest, combinedTrainTest
from plot_decision_regions import plot_decision_regions
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test = getStdXyTrainAndTest()
X_combined, y_combined, testidx_start, testidx_end = combinedTrainTest(
        X_train, y_train, X_test, y_test)

knn = KNeighborsClassifier(n_neighbors=5, p=2,
                           metric='minkowski')
knn.fit(X_train, y_train)

plot_decision_regions(X_combined, y_combined,
                      classifier=knn, 
                      test_idx=range(testidx_start, testidx_end))

plt.xlabel('petal length[std]')
plt.ylabel('petal width[std]')
plt.legend(loc='upper left')
plt.show()