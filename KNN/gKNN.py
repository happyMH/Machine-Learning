#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import random
import time
import math
import sklearn.preprocessing as preprocessing
import matplotlib.pyplot as plt

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

def gaussian(dist, a=1, b=0, c=0.3):
    return a * math.e ** (-(dist - b) ** 2 / (2 * c ** 2))

def distance(a,b):
    c = a
    for i in range(0,324):
        c[i] = (a[i] - b[i]) * (1 / (gaussian(a[i] - b[i]) + 1))

    dist = np.linalg.norm(c)         # 计算两个点的欧氏距离
    return dist

# 利用opencv获取图像hog特征
def get_hog_features(trainset):
    features = []

    hog = cv2.HOGDescriptor('hog.xml')

    for img in trainset:
        img = np.reshape(img,(28,28))
        cv_img = img.astype(np.uint8)

        hog_feature = hog.compute(cv_img)
        # hog_feature = np.transpose(hog_feature)
        features.append(hog_feature)

    features = np.array(features)
    features = np.reshape(features,(-1,324))

    return features

def Predict(testset,trainset,train_labels):
    predict = []
    count = 0

    for test_vec in testset:
        # 输出当前运行的测试用例坐标，用于测试
        print count
        count += 1

        knn_list = []       # 当前k个最近邻居
        max_index = -1      # 当前k个最近邻居中距离最远点的坐标
        max_dist = 0        # 当前k个最近邻居中距离最远点的距离

        # 先将前k个点放入k个最近邻居中，填充满knn_list
        for i in range(k):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离
            #dist = distance(train_vec,test_vec)

            knn_list.append((dist,label))

        # 剩下的点
        for i in range(k,len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]

            dist = np.linalg.norm(train_vec - test_vec)         # 计算两个点的欧氏距离
            #dist = distance(train_vec,test_vec)

            # 寻找10个邻近点钟距离最远的点
            if max_index < 0:
                for j in range(k):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]

            # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
            if dist < max_dist:
                knn_list[max_index] = (dist,label)
                max_index = -1
                max_dist = 0


        # 统计选票
        class_total = 10
        class_count = [0 for i in range(class_total)]
        sum_w = 0
        for dist,label in knn_list:
            sum_w += gaussian(dist)

        for dist,label in knn_list:
            class_count[label] += gaussian(dist) / sum_w        

        # 找出最大选票
        mmax = max(class_count)

        # 找出最大选票标签
        for i in range(class_total):
            if mmax == class_count[i]:
                predict.append(i)
                break

    return np.array(predict)

k = 8

if __name__ == '__main__':

    print 'Start read data'
    result = []
    size = []

    time_1 = time.time()

    train_data = np.fromfile("processed_mnist_train_data_2",dtype=np.uint8)
    train_labels = np.fromfile("mnist_train_label",dtype=np.uint8)

    test_data = np.fromfile("processed_mnist_test_data_2",dtype=np.uint8)
    test_labels = np.fromfile("mnist_test_label",dtype=np.uint8)

    train_features = train_data.reshape(60000,28,28)
    train_features = get_hog_features(train_features)

    test_features = test_data.reshape(10000,28,28)
    test_features = get_hog_features(test_features)

    time_2 = time.time()
    print 'read data cost ',time_2 - time_1,' second','\n'

    print 'Start training'
    print 'knn do not need to train'
    time_3 = time.time()
    print 'training cost ',time_3 - time_2,' second','\n'

    print 'Start predicting'
    test_predict = Predict(test_features,train_features,train_labels)
    time_4 = time.time()
    print 'predicting cost ',time_4 - time_3,' second','\n'

    score = accuracy_score(test_labels,test_predict)
    print "The accruacy socre is ", score
