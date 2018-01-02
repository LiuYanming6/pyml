# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:32:22 2017

@author: liuyanming
"""

from sklearn import datasets
#from sklearn.cross_validation import train_test_split will be removed in 0.20.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def getIris():
    # 提取数据
    iris = datasets.load_iris()
    X = iris.data[:, [2,3]]
    y = iris.target
    return train_test_split(
                X, y, test_size=0.3, random_state=0)

def getStdXyTrainAndTest():
    # 2 随机将数据矩阵按 7比3 划分为训练数据集和测试数据集
    # 适用于小规模数据
    X_train, X_test, y_train, y_test = getIris()

    '''特征缩放之归一化 normaliziation'''
    mms = MinMaxScaler()
    X_train_norm = mms.fit_transform(X_train)
    X_test_norm = mms.transform(X_test)

    '''特征缩放之标准化'''
    sc = StandardScaler()
    sc.fit(X_train) # 计算训练数据中的每个特征的样本均值和便准差
    X_train_std = sc.transform(X_train)
    X_test_std =  sc.transform(X_test) # 使用相同的缩放参数，保证和训练集彼此相当
    return X_train_std, X_test_std, y_train, y_test

def combinedTrainTest(X_train, y_train, X_test, y_test):
    X_combined = np.vstack((X_train, X_test))
    y_combined = np.hstack((y_train, y_test))
    print(len(y_train), len(y_combined))
    return X_combined, y_combined, len(y_train), len(y_combined)
