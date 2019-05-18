"""
This python script describes that how to use TensorFlow to implement the Kernel Support Vector Machine (SVM) model
for Multi-class Classification. The Kernel SVM model is embodied in defined functions about how to calculate the
overall loss.

Author:
    Hailiang Zhao
"""
import tensorflow as tf
import numpy as np
from sklearn.decomposition import PCA
from tensorflow.examples.tutorials.mnist import input_data


# ['define' necessary learning flags (batch size, step numbers, learning rate and train or test)]
tf.flags.DEFINE_integer('batch_size', 50, 'Number of samples per batch.')
tf.flags.DEFINE_integer('num_steps', 1000, 'Number of steps for training.')
tf.flags.DEFINE_integer('log_steps', 50, 'Number of steps per each display.')
tf.flags.DEFINE_boolean('is_evaluation', True, 'Whether or not the model should be evaluated.')
tf.flags.DEFINE_float('initial_learning_rate', 0.01, 'The initial learning rate for optimization.')
tf.flags.DEFINE_integer('num_classes', 10, 'The default number of classes of MNIST dataset.')

# ['define' flags used in Kernel SVM model]
tf.flags.DEFINE_float('gamma', -15.0, 'Penalty parameter of the error term.')

FLAGS = tf.flags.FLAGS


def prepare_label(labels_one_hot):
    """
    Encoding the original one hot label of pictures with '-1 & 1'.
    e.g.,
    [[0,1,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0]] -> [-1,1,-1,-1,-1,-1,-1,-1,-1,-1],[-1,-1,1,-1,-1,-1,-1,-1,-1,-1]]

    :param labels_one_hot: the vector of original labels
    :return: the encoded labels (numpy matrix with size being [10, num_samples])
    """
    labels = labels_one_hot
    labels[labels == 0] = -1
    labels = np.transpose(labels)
    return labels


def obtain_kernel(x_data, gamma):
    """
    Generate the RBF kernel.

    :param x_data: the input dataset
    :param gamma: the hyper parameter
    :return: the RBF kernel
    """
    square_distance = tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))
    return tf.exp(tf.multiply(gamma, tf.abs(square_distance)))


def obtain_cross_class_label(label):
    label_class_i = tf.reshape(label, [FLAGS.num_classes, 1, FLAGS.batch_size])
    label_class_j = tf.reshape(label_class_i, [FLAGS.num_classes, FLAGS.batch_size, 1])
    return tf.matmul(label_class_j, label_class_i)


def obtain_loss(alpha, label, kernel):
    """
    Calculate the overall loss.

    :param alpha: the parameter of dual problem
    :param label: the label of all dataset ([10, num_samples])
    :param kernel: the generated RBF kernel
    :return: the overall loss
    """
    term_1 = tf.reduce_sum(alpha)
    alpha_cross = tf.matmul(tf.transpose(alpha), alpha)
    cross_class_label = obtain_cross_class_label(label)
    term_2 = tf.reduce_sum(tf.multiply(kernel, tf.multiply(alpha_cross, cross_class_label)), [1, 2])
    return tf.reduce_sum(tf.subtract(term_2, term_1))


def predict_kernel(x_data, prediction_grid):
    A = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    B = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1), [-1, 1])
    square_distance = tf.add(tf.subtract(A, tf.multiply(2., tf.matmul(x_data, tf.transpose(prediction_grid)))),
                             tf.transpose(B))
    return tf.exp(tf.multiply(gamma, tf.abs(square_distance)))


def generate_batch(X, y, batch_size):
    idx = np.random.choice(len(X), size=batch_size)
    X_batch = X[idx]
    y_batch = y[:, idx]
    return X_batch, y_batch


if __name__ == '__main__':
    # load MNIST dataset
    mnist = input_data.read_data_sets("../../../dataset/MNIST/compressed", reshape=True, one_hot=True)

    # prepare labels ['define' labels of dataset]
    y_train = prepare_label(mnist.train.labels)
    y_test = prepare_label(mnist.test.labels)

    # use Principle Component Analysis (PCA) to compress dimension
    pca = PCA(n_components=100)
    pca.fit(mnist.train.images)
    print('The variance of the chosen components: %{0:.2f}'.format(100 * np.sum(pca.explained_variance_ratio_)))
    # prepare dataset ['define' feature of dataset]
    x_train = pca.transform(mnist.train.images)
    x_test = pca.transform(mnist.test.images)
    num_features = x_train.shape[1]

    # ['define' placeholders and variables/constants]
    data_placeholder = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
    label_placeholder = tf.placeholder(shape=[FLAGS.num_classes, None], dtype=tf.float32)
    predict_placeholder = tf.placeholder(shape=[None, num_features], dtype=tf.float32)
    # alpha is used to solve the dual problem
    alpha = tf.Variable(tf.random_normal(shape=[FLAGS.num_classes, FLAGS.batch_size]))
    # gamma is used to calculate the RBF kernel
    gamma = tf.constant(FLAGS.gamma)

    # calculation based on defined variables ['define' variables]
    kernel = obtain_kernel(data_placeholder, gamma)
    loss = obtain_loss(alpha, label_placeholder, kernel)
    predicted_kernel = predict_kernel(data_placeholder, predict_placeholder)

    # make prediction and test based on defined variables ['define' variables]
    output = tf.matmul(tf.multiply(label_placeholder, alpha), predicted_kernel)
    predicted_label = tf.argmax(output - tf.expand_dims(tf.reduce_mean(output, 1), 1), 0)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_label, tf.argmax(label_placeholder, 0)), tf.float32))

    # ['define' optimizer (the train operation)]
    train_op = tf.train.AdamOptimizer(FLAGS.initial_learning_rate).minimize(loss)

    # every variables have been defined, now we start running!
    # start session (or create graph)
    sess = tf.Session()
    # ['run' the initialization]
    sess.run(tf.global_variables_initializer())

    for step in range(FLAGS.num_steps):
        X_batch, y_batch = generate_batch(x_train, y_train, FLAGS.batch_size)
        # ['run' the optimization of this iteration]
        sess.run(train_op, feed_dict={data_placeholder: X_batch, label_placeholder: y_batch})
        # ['run' the calculation of loss and accuracy]
        loss_step = sess.run(loss, feed_dict={data_placeholder: X_batch, label_placeholder: y_batch})
        train_acc_step = sess.run(accuracy, feed_dict={data_placeholder: X_batch, label_placeholder: y_batch,
                                                       predict_placeholder: X_batch})
        X_batch_test, y_batch_test = generate_batch(x_test, y_test, FLAGS.batch_size)
        test_acc_step = sess.run(accuracy, feed_dict={data_placeholder: X_batch_test,
                                                      label_placeholder: y_batch_test,
                                                      predict_placeholder: X_batch_test})
        if step % FLAGS.log_steps == 0:
            print('Step %d, loss = %.2f, training_accuracy = %.2f %%, testing accuracy = %.2f %%' %
                  (step, loss_step, float(100 * train_acc_step), float(100 * test_acc_step)))
