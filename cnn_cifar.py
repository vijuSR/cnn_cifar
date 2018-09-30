import tensorflow as tf
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


class CifarLoader(object):

    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None

    def load(self):
        data = [unpickle(f) for f in self._source]
        images = np.vstack([d[bytes('data', encoding='utf8')] for d in data])
        n = len(images)

        self.images = images.reshape(n, 3, 32, 32).transpose(
            0, 2, 3, 1).astype(float) / 255

        self.labels = one_hot(
            np.hstack([d[bytes('labels', encoding='utf8')] for d in data]), 
            10
        )

        return self


    def next_batch(self, batch_size):
        
        x, y = self.images[self._i:self._i+batch_size], self.labels[self._i:self._i+batch_size]

        self._i = (self._i + batch_size) % len(self.images)

        return x, y


class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(
            ["data_batch_{}".format(i) for i in range(1, 6)]
        ).load()

        self.test = CifarLoader(["test_batch"]).load()


def one_hot(vec, classes=10):
    n = len(vec)
    out = np.zeros((n, classes))
    out[range(n), vec] = 1
    return out

def unpickle(file):
    with open(os.path.join(DATA_PATH, file), 'rb') as f:
        d = pickle.load(f, encoding='bytes')

    return d


def display_cifar(images, size):

    n = len(images)

    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
        for i in range(size)])
    plt.imshow(im)

    plt.show()


def weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], 
        padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], 
        strides=[1,2,2,1], padding='SAME')


def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)


def full_layer(input, size):
    insize = int(input.get_shape()[1])
    W = weight_variable([insize, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


def model1(x, keep_prob):

    conv1 = conv_layer(x, shape=[5,5,3,32])
    conv1_pool = max_pool_2x2(conv1)

    conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])
    conv2_pool = max_pool_2x2(conv2)

    conv2_flat = tf.reshape(conv2_pool, shape=[-1, 8*8*64])

    full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
    full_1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob)

    y_conv = full_layer(full_1_drop, 10)

    return y_conv


def model2(x, keep_prob):
    C1, C2, C3 = 30, 50, 80
    F1 = 500
    conv1_1 = conv_layer(x, shape=[3, 3, 3, C1])
    conv1_2 = conv_layer(conv1_1, shape=[3, 3, C1, C1])
    conv1_3 = conv_layer(conv1_2, shape=[3, 3, C1, C1])
    conv1_pool = max_pool_2x2(conv1_3)
    conv1_drop = tf. nn. dropout(conv1_pool, keep_prob=keep_prob)
    
    conv2_1 = conv_layer(conv1_drop, shape=[3, 3, C1, C2])
    conv2_2 = conv_layer(conv2_1, shape=[3, 3, C2, C2])
    conv2_3 = conv_layer(conv2_2, shape=[3, 3, C2, C2])
    conv2_pool = max_pool_2x2(conv2_3)
    conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)
    
    conv3_1 = conv_layer(conv2_drop, shape=[3, 3, C2, C3])
    conv3_2 = conv_layer(conv3_1, shape=[3, 3, C3, C3])
    conv3_3 = conv_layer(conv3_2, shape=[3, 3, C3, C3])
    conv3_pool = tf.nn.max_pool(conv3_3, ksize=[1, 8, 8, 1], strides=[1, 8, 8, 1], padding='SAME' )
    conv3_flat = tf.reshape(conv3_pool, [- 1, C3])
    conv3_drop = tf.nn.dropout(conv3_flat, keep_prob=keep_prob)
    
    full1 = tf.nn.relu(full_layer(conv3_flat, F1))
    full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)
    
    y_conv = full_layer(full1_drop, 10)
    
    return y_conv


def test(cifar_data, sess):
    x = cifar_data.test.images.reshape(10, 1000, 32, 32, 3)
    y = cifar_data.test.labels.reshape(10, 1000, 10)

    acc = np.mean([sess.run(accuracy, feed_dict={X: x[i], Y: y[i], keep_prob: 1.0}) for i in range(10)])
    print('test-accuracy: {:.4}%'.format(acc*100))


if __name__ == '__main__':

    BATCH_SIZE = 100
    STEPS = 2001

    DATA_PATH = os.path.join(
        os.path.dirname(__file__), 
        'data', 
        'cifar-10-batches-py'
    )
    cifar = CifarDataManager()
    print("Number of train images: {}".format(len(cifar.train.images)))
    print("Number of train labels: {}".format(len(cifar.train.labels)))
    print("Number of test images: {}".format(len(cifar.test.images)))
    print("Number of test images: {}".format(len(cifar.test.labels)))
    images = d.train.images
    display_cifar(images, 10)

    X = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    Y = tf.placeholder(tf.float32, shape=[None, 10])

    keep_prob = tf.placeholder(tf.float32)

    Y_conv = model2(X, keep_prob)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Y_conv, labels=Y))

    train = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(Y_conv, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(STEPS):
            accr = []
            batch = cifar.train.next_batch(BATCH_SIZE)
            [_, acc] = sess.run([train, accuracy], feed_dict={X:batch[0], Y:batch[1], keep_prob:0.5})
            accr.append(acc)

            if i % 100 == 0:
                print('step: {} train-accuracy: {:.4}%'.format(i, np.mean(acc)*100))
                accr = []

        test(cifar, sess)
