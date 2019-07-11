# -*- coding: utf-8 -*-
"""
Created on Fri May 10 09:17:34 2019

@author: 34563
"""
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import numpy as np

#加载数据
data = pd.read_csv('C:/Users/34563/Desktop/WS_TestsStudy/WS_Iterms/K-Means/kmeans-master/data.csv',encoding='gbk')
#GBK全称《汉字内码扩展规范》（GBK即“国标”、“扩展”汉语拼音的第一个字母，英文名称：Chinese Internal Code Specification）3。GBK是采用单双字节变长编码，英文使用单字节编码，完全兼容ASCII字符编码，中文部分采用双字节编码。
train_x = data[['2019年国际排名','2018世界杯','2015亚洲杯']]
df = pd.DataFrame(train_x)  #可以不用，因为之前已经是pd.read_csv（）
#规范化
mm = preprocessing.MinMaxScaler()
train_x = mm.fit_transform(train_x)
#构建聚类模型
kmeans = KMeans(n_clusters=3)
kmeans.fit(train_x)
predict_y = kmeans.predict(train_x)
#合并聚类结果，插入到原数据中
result = pd.concat((data,pd.DataFrame(predict_y)),axis=1)
result.rename({0:'聚类'},axis=1,inplace=True)
print(result)