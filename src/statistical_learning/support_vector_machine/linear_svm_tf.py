"""
This python script describes that how to use TensorFlow to implement the Linear Support Vector Machine (SVM) model.

Author:
    Hailiang Zhao
"""
import tensorflow as tf
from sklearn import datasets
import numpy as np
import sys
import matplotlib.pyplot as plt


# 'define' necessary learning flags (batch size, step numbers, learning rate and train or test)
tf.flags.DEFINE_integer('batch_size', 32, 'Number of samples per batch.')
tf.flags.DEFINE_integer('step_num', 5000, 'Number of iterations of training.')
tf.flags.DEFINE_float('initial_learning_rate', 0.1, 'The initial learning rate for optimization.')
tf.flags.DEFINE_boolean('is_evaluation', True, 'Whether or not the model is evaluated.')

# 'define' flags used in Linear SVM model
tf.flags.DEFINE_float('C_param', 0.1, 'Penalty parameter of the target error term.')
tf.flags.DEFINE_float('Reg_param', 1.0, 'Penalty parameter of the regularization term.')
tf.flags.DEFINE_float('delta', 1.0, 'The margin defined in SVM.')

FLAGS = tf.flags.FLAGS


def obtain_loss(W, b, x_data, y_target):
    """
    Calculate the total loss on provided dataset x with corresponding y y_target.

    :param W: the weight parameter
    :param b: the bias parameter
    :param x_data: the input dataset x
    :param y_target: the corresponding value of dataset x
    :return: total loss
    """
    logits = tf.subtract(tf.matmul(x_data, W), b)
    # regularization term
    norm_term = tf.divide(tf.reduce_sum(tf.multiply(tf.transpose(W), W)), 2)
    # target error
    classification_loss = tf.reduce_mean(tf.maximum(0., tf.subtract(FLAGS.delta, tf.multiply(logits, y_target))))
    total_loss = tf.add(tf.multiply(FLAGS.C_param, classification_loss), tf.multiply(FLAGS.Reg_param, norm_term))
    return total_loss


def inference(W, b, x_data, y_target):
    """
    Obtain the accuracy by inference.

    :param W: the weight parameter
    :param b: the bias parameter
    :param x_data: the input dataset x
    :param y_target: the corresponding value of dataset x
    :return: the accuracy
    """
    prediction = tf.sign(tf.subtract(tf.matmul(x_data, W), b))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(prediction, y_target), tf.float32))
    return accuracy


def generate_batch(x_train, y_train, num_samples=FLAGS.batch_size):
    """
    Generate the batch for training.

    :param x_train: the input dataset x
    :param y_train: the corresponding value of dataset x
    :param num_samples: the number of samples in one batch
    :return: the batched dataset
    """
    indices = np.random.choice(len(x_train), size=num_samples)
    X_batch = x_train[indices]
    y_batch = np.transpose([y_train[indices]])
    return X_batch, y_batch


# pre-process of iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = np.array([1 if label == 0 else -1 for label in iris.target])

# generate indices for train and test sets randomly
split = np.random.choice(X.shape[0], X.shape[0], replace=False)
train_indices = split[0:int(0.5 * X.shape[0])]
test_indices = split[int(0.5 * X.shape[0]):]
# 'define' train and test dataset
x_train = X[train_indices]
y_train = y[train_indices]
x_test = X[test_indices]
y_test = y[test_indices]

# 'define' placeholders and variables
x_data = tf.placeholder(shape=[None, X.shape[1]], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
W = tf.Variable(tf.random_normal(shape=[X.shape[1], 1]))
b = tf.Variable(tf.random_normal(shape=[1, 1]))

# 'define' loss and accuracy (use the pre-declared functions)
total_loss = obtain_loss(W, b, x_data, y_target)
accuracy = inference(W, b, x_data, y_target)

# 'define' train operation
train_op = tf.train.GradientDescentOptimizer(FLAGS.initial_learning_rate).minimize(total_loss)

if __name__ == '__main__':
    # 'start' session
    sess = tf.Session()

    # 'run' initialization
    init = tf.global_variables_initializer()
    sess.run(init)

    # training the linear SVM
    for step in range(FLAGS.step_num):
        X_batch, y_batch = generate_batch(x_train, y_train, num_samples=FLAGS.batch_size)

        # 'run' the optimizer
        # the feed_dict are those variables in placeholders!
        sess.run(train_op, feed_dict={x_data: X_batch, y_target: y_batch})

        # 'run' the calculation of the loss and accuracy
        loss_step = sess.run(total_loss, feed_dict={x_data: X_batch, y_target: y_batch})
        train_acc_step = sess.run(accuracy, feed_dict={x_data: x_train, y_target: np.transpose([y_train])})
        test_acc_step = sess.run(accuracy, feed_dict={x_data: x_test, y_target: np.transpose([y_test])})

        # middle results output
        if step % 100 == 0:
            print('Step %d, training_accuracy = %.2f %%, testing accuracy = %.2f %%' %
                  (step, float(100 * train_acc_step), float(100 * test_acc_step)))

    # show the training results with figure
    if FLAGS.is_evaluation:
        [[w1], [w2]] = sess.run(W)
        [[bias]] = sess.run(b)
        x_line = [data[1] for data in X]

        # find the separator line
        line = [-w2 / w1 * i + bias / w1 for i in x_line]

        positive_X, positive_y = [], []
        negative_X, negative_y = [], []
        for idx, data in enumerate(X):
            if y[idx] == 1:
                positive_X.append(data[1])
                positive_y.append(data[0])
            elif y[idx] == -1:
                negative_X.append(data[1])
                negative_y.append(data[0])
            else:
                sys.exit('Invalid label!')

        # plot the SVM decision boundary
        plt.plot(positive_X, positive_y, '+', label='Positive')
        plt.plot(negative_X, negative_y, 'o', label='Negative')
        plt.plot(x_line, line, 'r-', label='Separator', linewidth=3)
        plt.legend(loc='best')
        plt.title('Linear SVM')
        plt.savefig('linear_SVM.eps')
        plt.show()
        plt.close()
