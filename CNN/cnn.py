import numpy as np
import tensorflow as tf
from datetime import datetime
import os
from prepare import read_data

class dataset(object):
    def __init__(self, data):
        self.data = data
        self.size = data.shape[0]
        self.now = 0
        
    def next_batch(self, n):
        d = self.data[self.now:self.now+n]
        self.now += n
        if self.now == self.size:
            self.now = 0
        return d

class mnist_data(object):  # a wrapper for mnist data
    def __init__(self):
        train_data, train_label, test_data, test_label = read_data(2)
        train_data.astype('float16')
        train_data = train_data/255
        test_data.astype('float16')
        test_data = test_data/255
        self.train_data = dataset(train_data)
        self.train_label = dataset(self.modify_label(train_label))
        self.test_data = dataset(test_data)
        self.test_label = dataset(self.modify_label(test_label))

    def modify_label(self,data): #  one-hot encoding
        n = data.shape[0]
        new_data = np.zeros((n,10), dtype=np.uint8)
        for i in range(n):
            new_data[i][data[i]] = 1
        return new_data

    def next_train(self, n):
        return (self.train_data.next_batch(n), self.train_label.next_batch(n))

    def next_test(self, n):
        return (self.test_data.next_batch(n), self.test_label.next_batch(n))

def predict(accuracy):
    print("Predicting...")
    result = list()
    for i in range(100):
        batch = mnist.next_test(100)
        result.append(accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0}))
    print("Accuracy: %g" % np.mean(result))
    return np.mean(result)

def write_log(var, log_name):
    path = os.path.join("logs", log_name)
    with open(path, "w") as fout:
        for value in var:
            fout.write(str(value))
            fout.write('\n')

def model():
    x = tf.placeholder("float", [None, 784])  # input
    y_ = tf.placeholder("float", [None, 10])  # output
    x_image = tf.reshape(x, [-1, 28, 28, 1])
    # the first convolutinal layer
    conv1 = tf.layers.conv2d(
      inputs=x_image,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    # the first pooling layer
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    # the second convolutinal layer
    conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
    # the second pooling layer
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    # fully connected layer
    pool2_flat = tf.reshape(pool2, [-1, 7*7*64])
    fc = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    # dropout layer
    keep_prob = tf.placeholder("float")
    dropout = tf.layers.dropout(inputs=fc,rate=keep_prob)
    #dropout = fc
    # fully connected layer, output layer
    output = tf.layers.dense(inputs=dropout, units=10, activation=tf.nn.softmax)

    # train and evaluate the model
    cross_entropy = -tf.reduce_sum(y_*tf.log(output))
    train_step = tf.train.GradientDescentOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy, cross_entropy, train_step, x, y_, keep_prob
    
def train(accuracy, cross_entropy, train_step, x, y_, keep_prob):
    print("Training...")
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    
    result = []
    train_loss = []
    test_loss = []

    for loop in range(5):
        for j in range(600):
            i = loop * 600 + j
            batch = mnist.next_train(100)
            if i%100 == 0:
                #train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
                #print("step %d, train accuracy %g" %(i, train_accuracy))
                loss = cross_entropy.eval(feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
                train_loss.append(loss)
                print("step %d, train loss %g" %(i, loss))
                testbatch = mnist.next_test(100)
                test_loss.append(cross_entropy.eval(feed_dict={x:testbatch[0], y_:testbatch[1], keep_prob:1.0}))
            train_step.run(feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})
        result.append(predict(accuracy))

    write_log(result, "cnn_predict.txt")
    write_log(train_loss, "cnn_train_loss.txt")
    write_log(test_loss, "cnn_test_loss.txt")

mnist = mnist_data()
start = datetime.now()
accuracy, cross_entropy, train_step, x, y_, keep_prob = model()
train(accuracy, cross_entropy, train_step, x, y_, keep_prob)
finish = datetime.now()
print(finish-start)
