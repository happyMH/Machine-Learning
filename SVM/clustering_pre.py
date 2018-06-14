import numpy as np
import csv

##读入数据集
data = np.fromfile("processed_mnist_train_data_2",dtype=np.uint8)
label = np.fromfile("mnist_train_label",dtype=np.uint8)
data = data.reshape(60000,28*28)

label_0 = []
label_1 = []
label_2 = []
label_3 = []
label_4 = []
label_5 = []
label_6 = []
label_7 = []
label_8 = []
label_9 = []

##统计每个label的数据
for i in range(0,60000):
    if label[i] == 0: label_0.append(data[i])
    elif label[i] == 1: label_1.append(data[i])
    elif label[i] == 2: label_2.append(data[i])
    elif label[i] == 3: label_3.append(data[i])
    elif label[i] == 4: label_4.append(data[i])
    elif label[i] == 5: label_5.append(data[i])
    elif label[i] == 6: label_6.append(data[i])
    elif label[i] == 7: label_7.append(data[i])
    elif label[i] == 8: label_8.append(data[i])
    elif label[i] == 9: label_9.append(data[i])
    
pre_data = []  
pre_data.append(label_0)
pre_data.append(label_1)
pre_data.append(label_2)
pre_data.append(label_3)
pre_data.append(label_4)
pre_data.append(label_5)
pre_data.append(label_6)
pre_data.append(label_7)
pre_data.append(label_8)
pre_data.append(label_9)

from sklearn.cluster import AgglomerativeClustering
##可以修改该值设置聚类数目
clusters=130
##可以设置该值设置从每类中提取的样本数目
samp_num=2

final_data = []
index = 0
num = 0
for i in pre_data:
    print "processing",index,"..."
    model = AgglomerativeClustering(n_clusters=clusters).fit(np.array(i))
    labels = model.labels_
    for j in range(0,clusters):
        num = 0
        for t in range(0,len(labels)):
            if labels[t] == j:
                num = num + 1
                tmp = list(i[t])
                tmp.insert(0,index) #加入label
                final_data.append(np.array(tmp))
                if num == samp_num:
                    break
    index = index + 1

##将得到的数据集写入.CSV文件
with open("test_140_2.csv","wb") as csvfile: 
    writer = csv.writer(csvfile)
    writer.writerows(final_data)
    


    
                



