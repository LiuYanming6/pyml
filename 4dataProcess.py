#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:36:37 2017

@author: liuyanming

数据预处理

"""

import pandas as pd
from io import StringIO
import numpy as np

'''1. 数值型的
'''
csv_data = '''A, B, C, D
1.0, 2.0, 3.0,4.0
5.0, 6.0,,8
0, 11,12
'''

df = pd.read_csv(StringIO(csv_data))
print(df)

# 删除有缺失值的特征
df.dropna()

# 缺失值填充  mean-特征列的均值填充
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
imr.fit(df)
imputed_data = imr.transform(df.values)
print(imputed_data)



"""2. 类别数据处理
color nominal feature 标称特征
size  ordinal feature 有序特征
"""
df = pd.DataFrame([
        ['green', 'M', 10.1, 'class1'],
        ['red', 'L', 13.5, 'class2'],
        ['blue', 'XL', 15.3, 'class1']
        ])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)

# 有序特征映射 及逆映射
size_mapping = {
        'XL': 3,
        'L' : 2,
        'M' : 1}
df['size'] = df['size'].map(size_mapping)
print(df)

# 如果要还原回去
#inv_size_mapping = {v:k for k, v in size_mapping.items()}
#df['size'] = df['size'].map(inv_size_mapping)
#print(df)

# 标签的编码 方法1
class_mapping = {label: idx for idx, label in enumerate(np.unique(df['classlabel']))}
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)

# 方法2 更方便
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)

y1 = class_le.inverse_transform(y)
print(y1)

# 颜色这个变量可以像处理标签的方法，但数值有大小，这样处理的结果不是最优的
# one-hot encoding 独热编码技术，创建一个新的虚拟特征(dummy feature)
X = df[['color', 'size', 'price']].values
X[:, 0] = class_le.fit_transform(X[:, 0])
print(X)

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0]) # default sparse=Ture
print(ohe.fit_transform(X).toarray()) # 返回一个厂规数组
#print(ohe.fit_transform(X)) 什么是稀疏矩阵 sparse matrix

# or 可以省略 80 - 87 好多行
#data = pd.get_dummies(df[['color', 'size', 'price']])


