"""
This python script describes that how to use TensorFlow to implement the Linear Regression model.

Author:
    Hailiang Zhao
"""
import os
import tensorflow as tf
import xlrd
import numpy as np
import matplotlib.pyplot as plt

# remove the AVX warning
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# read dataset from xls file
DATA_FILE = '../dataset/fire_theft.xls'
book = xlrd.open_workbook(DATA_FILE, encoding_override='utf-8')
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
num_samples = sheet.nrows - 1

tf.app.flags.DEFINE_integer(
    'num_epochs', 10, 'The number of epochs for training the model, 50 in default.'
)
FLAGS = tf.app.flags.FLAGS

# create weight and bias
W = tf.Variable(0.0, name='weight')
b = tf.Variable(0.0, name='bias')


def inputs():
    """
    Define the place holders of features and y.

    :return: the dataset and y place holders
    """
    x = tf.placeholder(tf.float32, name='X')
    y = tf.placeholder(tf.float32, name='Y')
    return x, y


def inference(X):
    """
    Pass the features of dataset then return the y.

    :param X: the input features
    :return: W * x + b
    """
    return W * X + b


def loss(X, Y):
    """
    Compute the loss by comparing the predicted value and the true y.

    :param X: the input features
    :param Y: the true y
    :return: the squared loss
    """
    Y_predicted = inference(X)
    return tf.squared_difference(Y, Y_predicted)


def train(loss):
    """
    Use gradient Descent Method to train the model parameter w.

    :return: the optimizer who minimizes the squared loss
    """
    learning_rate = 0.0001
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)


if __name__ == '__main__':
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # get the input tensors
        X, Y = inputs()

        # return the train loss and create the train op
        train_loss = loss(X, Y)
        train_op = train(train_loss)

        # train the model
        for epoch_num in range(FLAGS.num_epochs):
            for x, y in data:
                # update train_op in every epoch
                train_op = train(train_loss)

                # run train_op in sess
                loss_value, _ = sess.run([train_loss, train_op], feed_dict={X: x, Y: y})
            print('epoch %d, loss = %d' % (epoch_num + 1, loss_value))
            # update the values of weight and bias
            final_weight, final_bias = sess.run([W, b])

    # evaluate and plot the final results
    input_values = data[:, 0]
    labels = data[:, 1]
    predict_values = data[:, 0] * final_weight + final_bias
    plt.plot(input_values, labels, 'ro', label='true')
    plt.plot(input_values, predict_values, label='predicted')
    plt.legend()
    plt.savefig('linear_regression.eps')
    plt.show()
    plt.close()
