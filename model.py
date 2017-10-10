from PIL import Image
import tensorflow as tf
import numpy as np
import os


class Model:
    """ The neural network for recognizing captcha """
    def __init__(self, train_dir, test_dir):
        self.sess = tf.Session()
        self.train_data, self.train_labels = self.load_data(train_dir)
        self.test_data, self.test_labels = self.load_data(test_dir)
        self.x = tf.placeholder(tf.float32, [None, len(self.train_data[0])])
        self.W = tf.Variable(tf.zeros([len(self.train_data[0]), 10]))
        self.b = tf.Variable(tf.zeros([10]))
        self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        self.y_ = tf.placeholder(tf.float32, [None, 10])

    def next_batch(self, n):
        idx = np.random.randint(0, len(self.train_data), n)
        return [self.train_data[i] for i in idx], [self.train_labels[j] for j in idx]

    @staticmethod
    def load_data(dir_name):
        data = []
        labels = []
        files = os.listdir(dir_name)
        for name in files:
            fullname = os.path.join(dir_name, name)
            im = Image.open(fullname)
            im = im.convert('1')
            im_data = np.array(im, dtype='int').flatten()
            extended = np.zeros(16 * 16, dtype='int')
            extended[0:im_data.shape[0]] = im_data
            data.append(extended)

            classes = np.zeros(10)
            f_name = name.replace('.jpg', '')
            classes[int(f_name[len(f_name) - 1])] = 1.0
            labels.append(classes)
        return data, labels

    def train(self, epoch=1000):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(self.y_ * tf.log(self.y), reduction_indices=[1]))
        train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
        self.sess.run(tf.initialize_all_variables())
        for i in range(epoch):
            batch_xs, batch_ys = self.next_batch(20)
            self.sess.run(train_step, feed_dict={self.x: batch_xs, self.y_: batch_ys})
            correct = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            if i % 100 == 0:
                print("epoch %d: %f" % (i, self.sess.run(accuracy, feed_dict={self.x: self.test_data, self.y_: self.test_labels})))

    def predict(self, test_images):
        data = []
        for im in test_images:
            im_data = np.array(im, dtype='int').flatten()
            extended = np.zeros(16 * 16, dtype='int')
            extended[0:im_data.shape[0]] = im_data
            data.append(extended)
        p_class = tf.argmax(self.y, 1)
        return self.sess.run(p_class, feed_dict={self.x: data})
