"""
This module provides the implementation of AdaBoost algorithm without calling any pre-defined models.

Author:
    Hailiang Zhao
"""
import numpy as np
from src.utils.roc import plot_roc


def load_45data():
    """
    This function loads the famous 45 problem data that decision trees are notorious
    for having difficulty with.

    :return: data matrix and labels
    """
    data_arr = [[1.0, 2.1], [2.0, 1.1], [1.3, 1.0], [1.0, 1.0], [2.0, 1.0]]
    label_arr = [1.0, 1.0, -1.0, -1.0, 1.0]
    return data_arr, label_arr


def value_compare(data_mat, dim, threshold, unequal_type):
    """
    This function is used to test if any of values are less than or greater than the
    threshold value we're testing.

    :param data_mat:
    :param dim:
    :param threshold:
    :param unequal_type: greater than ('gt') or less than ('ls')
    :return:
    """
    predicted = np.ones((np.shape(data_mat)[0], 1))
    if unequal_type == 'lt':
        predicted[data_mat[:, dim] <= threshold] = -1.
    else:
        predicted[data_mat[:, dim] > threshold] = -1.
    return predicted


def build_decision_stump(data_arr, label_arr, weights, steps_num=10):
    """
    This function builds the decision stump classifier by looping over a weighted version
    of the dataset and finding the stump that yields the lowest error.
    The decision stump acts as the weak learner for our ensemble method.

    :param data_arr:
    :param label_arr:
    :param weights:
    :param steps_num:
    :return:
    """
    data_mat, label_mat = np.mat(data_arr), np.mat(label_arr).T
    m, n = np.shape(data_mat)
    best_stump = {}
    best_classified = np.mat(np.zeros((m, 1)))
    min_error = np.inf

    # for each dimension of the dataset
    for i in range(n):
        min_value, max_value = data_mat[:, i].min(), data_mat[:, i].max()
        step_size = (max_value - min_value) / steps_num
        for j in range(-1, int(steps_num)+1):
            for unequal_type in ['lt', 'gt']:
                threshold = min_value + float(j) * step_size
                predicted = value_compare(data_mat, i, threshold, unequal_type)
                errors = np.mat(np.ones((m, 1)))
                errors[predicted == label_mat] = 0
                weighted_error = weights.T * errors
                print('Split dim #%d, threshold: %.2f, unequal type: %s, the weighted error is %.3f' %
                      (i, threshold, unequal_type, weighted_error))
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_classified = predicted.copy()
                    best_stump['dim'] = i
                    best_stump['threshold'] = threshold
                    best_stump['unequal_type'] = unequal_type
    return best_stump, min_error, best_classified


def ada_boost(data_arr, label_arr, classifiers_num=40):
    """
    This function implements a simple AdaBoost algorithm with the weak classifier being decision stump.

    :param data_arr:
    :param label_arr:
    :param classifiers_num:
    :return:
    """
    weak_classifiers = []
    m = np.shape(data_arr)[0]
    instance_weights = np.mat(np.ones((m, 1)) / m)
    aggregate_classified = np.mat(np.zeros((m, 1)))

    for i in range(classifiers_num):
        stump, error, classified = build_decision_stump(data_arr, label_arr, instance_weights)
        print('Current weights of each instance:\n', instance_weights)

        # calculate the weight of current weak classifier
        classifier_weight = float(0.5 * np.log((1. - error) / max(error, 1e-16)))
        stump['weight'] = classifier_weight
        weak_classifiers.append(stump)
        print('Current weak classifier\'s classification result:\n', classified.T)

        # update the weight vector so that the instances that are correctly classified will decrease in weight and
        # the misclassified instances will increase in weight
        instance_weights = np.multiply(
            instance_weights,
            np.exp(
                np.multiply(-1*classifier_weight*np.mat(label_arr).T, classified)
            )
        )
        instance_weights /= instance_weights.sum()

        # update the overall classification result
        aggregate_classified += classifier_weight * classified
        print('Current aggregate classification result:\n', aggregate_classified.T)
        error_rate = np.multiply(np.sign(aggregate_classified) != np.mat(label_arr).T, np.ones((m, 1))).sum() / m
        print('Total training error:', error_rate*100, '%%\n')
        if error_rate == 0.:
            break
    # aggregate_classified is returned for plotting ROC
    return weak_classifiers, aggregate_classified


def classify(data_arr, weak_classifiers):
    """
    Classify new data instances with the trained cluster of weak classifiers.

    :param data_arr:
    :param weak_classifiers:
    :return:
    """
    data_mat = np.mat(data_arr)
    m = np.shape(data_mat)[0]
    aggregate_classified = np.mat(np.zeros((m, 1)))
    for i in range(len(weak_classifiers)):
        classified = value_compare(
            data_mat,
            weak_classifiers[i]['dim'],
            weak_classifiers[i]['threshold'],
            weak_classifiers[i]['unequal_type'])
        aggregate_classified += weak_classifiers[i]['weight'] * classified
    return np.sign(aggregate_classified)


def colic_test():
    train_fr = open('../../../dataset/logistic/horse_colic_train.txt')
    train_features, train_labels = [], []
    for line in train_fr.readlines():
        line_data = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(line_data[i]))
        train_features.append(line_arr)
        label = 1. if float(line_data[21]) == 1. else -1.
        train_labels.append(label)
    classifiers, aggregate_classified = ada_boost(train_features, train_labels)
    for i in range(len(classifiers)):
        print(classifiers[i])
    _ = plot_roc(aggregate_classified.T, train_labels)

    test_fr = open('../../../dataset/logistic/horse_colic_test.txt')
    test_num = 0
    error_count = 0
    test_features, test_labels = [], []
    for line in test_fr.readlines():
        line_data = line.strip().split('\t')
        line_arr = []
        for i in range(21):
            line_arr.append(float(line_data[i]))
        test_features.append(line_arr)
        label = 1. if float(line_data[21]) == 1. else -1.
        test_labels.append(label)
        test_num += 1
    test_results = classify(test_features, classifiers)
    for i in range(test_num):
        if test_results[i] != test_labels[i]:
            error_count += 1
    print('The testing error is: %f%%' % (float(error_count) / test_num))


if __name__ == '__main__':
    # simple test
    # data, label = load_45data()
    # weights = np.mat(np.ones((5, 1)) / 5)
    # stump, error, classified = build_decision_stump(data, label, weights)
    # print(stump)
    # classifiers, _ = ada_boost(data, label)
    # print(classifiers)

    # colic test
    colic_test()
