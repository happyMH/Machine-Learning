import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ����OpenCV��ȡͼ��hog����
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

##������Լ�
test_data = np.fromfile("processed_mnist_test_data_2",dtype=np.uint8)
test_label = np.fromfile("mnist_test_label",dtype=np.uint8)

test_data = test_data.reshape(10000,28*28)
test_label = test_label.reshape(-1,1)

##����ͨ��������ȡ��ѵ����
raw_data = pd.read_csv('test_160.csv',header=None)
data = raw_data.values
##�õ�������label
imgs = data[::,1::]
labels = data[::,0]

##ȥ����������ע�ͽ�ʹ��hog����
##imgs = get_hog_features(imgs)
##
##test_data = get_hog_features(test_data)

from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
svm = OneVsOneClassifier(SVC(kernel='linear'))
print "fitting..."
svm.fit(imgs,labels)

print "SVM model's Score:  ",
print svm.score(test_data,test_label)
print

