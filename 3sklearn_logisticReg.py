# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:22:23 2017

@author: liuyanming
"""

from sklearn_datasets import getStdXyTrainAndTest, combinedTrainTest
X_train_std, X_test_std, y_train, y_test = getStdXyTrainAndTest()

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)


X_combined, y_combined, testidx_start, testidx_end = combinedTrainTest(X_train_std, y_train, X_test_std, y_test)
from plot_decision_regions import plot_decision_regions
plot_decision_regions(X_combined, y_combined, lr,
                      range(0, testidx_end))

import matplotlib.pyplot as plt
plt.xlabel('petal len(std) ')
plt.ylabel('petal width(std)')
plt.legend(loc='upper left')
plt.show()


"""
基于RBF核的SVM
"""
from sklearn.svm import SVC
from plot_decision_regions import plot_decision_regions
svm = SVC(kernel='rbf', random_state=0, gamma=.1, C=10.0)
svm.fit(X_combined, y_combined)
plot_decision_regions(X_combined, y_combined, classifier=svm)
plt.legend(loc='upper left')
plt.show()

