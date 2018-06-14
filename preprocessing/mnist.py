from prepare import read_data
import os
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals.joblib import load, dump
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.decomposition import PCA

train_data_num = 60000  # The number of training figures
test_data_num = 10000  # The number of testing figures
fig_w = 45  # width of each figure

def accuracy(test_label, predict_label):  # accuracy of the model
    total = len(test_label)
    correct = np.sum(test_label==predict_label)
    return correct/total

def save_model(model, file_name):
    dump(model, file_name, compress=3)

def load_model(file_name):
    return load(file_name)

def load_predict(file_name, test_data, test_label):  # load a model and then use ti to predict
    file_name = os.path.join("model", file_name)
    print("Loading model...")
    clf = load_model(file_name)
    print("Predicting...")
    predict_label = clf.predict(test_data)
    acc = accuracy(test_label, predict_label)
    print("Accuracy: ", acc)
    exit(0)
    for i in range(test_data_num):
        if predict_label[i] != test_label[i]:
            print(i)
    exit(0)

def SVM(train_data, train_label):
    #train_data = train_data[:1000]
    #train_label = train_label[:1000]
    #pca = PCA(n_components=100)
    #train_data = pca.fit_transform(train_data)
    #print(pca.explained_variance_ratio_)
    #print(sum(pca.explained_variance_ratio_))
    #exit(0)
    clf = svm.SVC(kernel='linear', verbose=1)
    clf.fit(train_data, train_label)
    return clf

def lr(train_data, train_label):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(penalty='l2', tol=0.001, verbose=1)
    clf.fit(train_data, train_label)
    return clf

def mlp(train_data, train_label):
    layer = tuple([100 for i in range(15)])
    clf = MLPClassifier(hidden_layer_sizes=layer, verbose=1)
    clf.fit(train_data, train_label)
    return clf

def knn(train_data, train_label):
    from knn import knn
    clf = knn(10)
    clf.fit(train_data, train_label)
    return clf

def m3(test_data):
    file_name1 = os.path.join("model", "svm6")
    file_name2 = os.path.join("model", "mlp3")
    file_name3 = os.path.join("model", "lr1")
    print("Loading model...")
    clf1 = load_model(file_name1)
    clf2 = load_model(file_name2)
    clf3 = load_model(file_name3)
    print("Predicting...")
    predict_label1 = clf1.predict(test_data)
    predict_label2 = clf2.predict(test_data)
    predict_label3 = clf3.predict(test_data)
    print("")
    predict_label = []
    for i in range(test_data_num):
        if predict_label1[i] == predict_label2[i]:
            predict_label.append(predict_label1[i])
        elif predict_label1[i] == predict_label3[i]:
            predict_label.append(predict_label1[i])
        elif predict_label2[i] == predict_label3[i]:
            predict_label.append(predict_label2[i])
        else:
            predict_label.append(predict_label2[i])
    return predict_label
    
def main():
    train_data, train_label, test_data, test_label = read_data("HOG")
    #load_predict("mlp3", test_data, test_label)
    print("Fitting...")
    clf = SVM(train_data, train_label)
    save_model(clf, "svm7")
    print("Predicting...")
    #pca = PCA(n_components=100)
    #test_data = pca.fit_transform(test_data)
    #test_data = test_data[:300]
    #test_label = test_label[:300]
    predict_label = clf.predict(test_data)
    #predict_label = m3(test_data)
    acc = accuracy(test_label, predict_label)
    print("Accuracy: ", acc)

if __name__=="__main__":
    start_time = datetime.now()
    main()
    finish_time = datetime.now()
    print()
    print("Start: ", start_time)
    print("Finish: ", finish_time)
    print("Time cost: ", finish_time-start_time)
