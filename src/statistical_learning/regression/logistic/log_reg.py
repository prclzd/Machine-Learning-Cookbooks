"""
This module provides the Logistic Regression algorithm without call any pre-defined models.

Author:
    Hailiang Zhao
"""
import numpy as np
import matplotlib.pyplot as plt
import random


def load_dataset(filename='../../../dataset/logistic/test_set.txt'):
    features, labels = [], []
    fr = open(filename)
    for line in fr.readlines():
        line_data = line.strip().split()
        features.append([1., float(line_data[0]), float(line_data[1])])
        labels.append(int(line_data[2]))
    return features, labels


def sigmoid(x):
    return 1. / (1 + np.exp(-x))


def gradient_ascent(features, labels, learning_rate=0.001, iterations=500):
    # numpy matrix with size m \times n
    features_matrix = np.mat(features)
    # numpy matrix with size m \times 1
    labels_matrix = np.mat(labels).transpose()
    m, n = np.shape(features_matrix)
    weights = np.ones((n, 1))
    for k in range(iterations):
        predicted = sigmoid(features_matrix * weights)
        errors = labels_matrix - predicted
        weights = weights + learning_rate * features_matrix.transpose() * errors
    return weights


def stochastic_gradient_ascent(features, labels, learning_rate=0.01):
    features_matrix = np.array(features)
    m, n = np.shape(features_matrix)
    weights = np.ones(n)
    # actually it's not 'stochastic', it's on sequence
    # every time we update weights based on only one instance
    # the iteration num is the number of instances
    for i in range(m):
        predicted = sigmoid(sum(features_matrix[i] * weights))
        error = labels[i] - predicted
        weights = weights + learning_rate * error * features_matrix[i]
    return weights


def updated_sto_grad_scent(features, labels, iterations=150):
    features_matrix = np.array(features)
    m, n = np.shape(features_matrix)
    weights = np.ones(n)
    for j in range(iterations):
        indices = list(range(m))
        for i in range(m):
            learning_rate = 4 / (1. + j + i) + 0.01
            rand_idx = int(random.uniform(0, len(indices)))
            predicted = sigmoid(sum(features_matrix[rand_idx] * weights))
            error = labels[rand_idx] - predicted
            weights = weights + learning_rate * error * features_matrix[rand_idx]
            del(indices[rand_idx])
    return weights


def classify(feature, weights):
    """
    Classification by Logistic Regression.

    :param feature:
    :param weights:
    :return:
    """
    if sigmoid(sum(feature * weights)) > 0.5:
        return 1.
    else:
        return 0.


def plot_best_fit(weights_arr, features, labels):
    features_arr = np.array(features)
    point_num = np.shape(features_arr)[0]
    x1, y1 = [], []
    x2, y2 = [], []
    for i in range(point_num):
        if int(labels[i]) == 1:
            x1.append(features_arr[i, 1])
            y1.append(features_arr[i, 2])
        else:
            x2.append(features_arr[i, 1])
            y2.append(features_arr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='blue')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights_arr[0] - weights_arr[1] * x) / weights_arr[2]
    ax.plot(x, y)
    plt.xlabel(r'$x_1$')
    plt.ylabel(r'$x_2$')
    plt.show()


def colic_test():
    train_fr = open('../../../../dataset/logistic/horse_colic_train.txt')
    test_fr = open('../../../../dataset/logistic/horse_colic_test.txt')
    train_features, train_labels = [], []
    for line in train_fr.readlines():
        line_data = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(line_data[i]))
        train_features.append(line_arr)
        train_labels.append(float(line_data[21]))
    up_sto_grad_weights = updated_sto_grad_scent(np.array(train_features), train_labels, 500)

    error_counts = 0.
    test_ins_num = 0
    for line in test_fr.readlines():
        test_ins_num += 1
        line_data = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(line_data[i]))
        if int(classify(np.array(line_arr), up_sto_grad_weights)) != int(line_data[21]):
            error_counts += 1
    error_rate = float(error_counts) / test_ins_num
    print('The error rate is: %f%%' % (error_rate * 100))
    return error_rate


if __name__ == '__main__':
    data, label = load_dataset()
    grad_weights = gradient_ascent(data, label)
    plot_best_fit(grad_weights.getA(), data, label)

    sto_grad_weights = stochastic_gradient_ascent(data, label)
    plot_best_fit(sto_grad_weights, data, label)

    mo_sto_grad_weights = updated_sto_grad_scent(data, label)
    plot_best_fit(mo_sto_grad_weights, data, label)

    colic_test()
