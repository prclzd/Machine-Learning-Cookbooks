"""
This module defines k-Nearest-Neighbors (kNN) algorithm without call any pre-defined models.
Examples: dating_test(); handwriting_recognition().

Author:
    Hailiang Zhao
"""
import numpy as np
import operator
import matplotlib.pyplot as plt
import os


def classify(input_feature, features, labels, k):
    """
    This function implements the kNN algorithm by comparing the input_feature and
    feature of all instances in dataset.

    :param input_feature: the features of the input instance
    :param features: the features of instances in dataset
    :param labels: the labels of instances in dataset
    :param k: the number of neighbors
    :return: the final voted label
    """
    dataset_size = features.shape[0]
    diff_mat = np.tile(input_feature, (dataset_size, 1)) - features
    distances = ((diff_mat**2).sum(axis=1))**0.5
    sorted_indices = distances.argsort()
    class_count = {}
    for i in range(k):
        vote = labels[sorted_indices[i]]
        class_count[vote] = class_count.get(vote, 0) + 1
    sorted_class = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class[0][0]


def txt2matrix(filename='../../../dataset/dating/datingTestSet2.txt', separator='\t', needed_features_num=3):
    """
    Read txt file and convert the data into numpy array.

    :param filename: the filename (directory) where dating.txt stores
    :param separator: the separator of data in the txt file
    :param needed_features_num: the number of needed features
        --> feature1: percentage of time spent playing video games
        --> feature2: number of frequent flying miles earned per year
        --> feature3: liters of ice cream consumed weekly
        --> category: 'didntLike':1, 'smallDoses': 2, 'largeDoses': 3
    :return: numpy matrix for features and list of corresponding labels
    """
    fr = open(filename)
    return_mat = np.zeros((len(fr.readlines()), needed_features_num))
    fr.close()

    return_labels = []
    fr = open(filename)
    index = 0
    for l in fr.readlines():
        line = l.strip()
        list_from_line = line.split(separator)
        return_mat[index, :] = list_from_line[0:needed_features_num]
        return_labels.append(int(list_from_line[-1]))
        index += 1
    fr.close()
    return return_mat, return_labels


def auto_norm(features):
    """
    Obtain the normed features by data-normalizing.

    :param features: the features of instances in dataset
    :return: the normed features
    """
    # get the min and max value of each column
    min_values = features.min(0)
    max_values = features.max(0)
    ranges = max_values - min_values
    normed_features = features - np.tile(min_values, (features.shape[0], 1))
    normed_features = normed_features / np.tile(ranges, (features.shape[0], 1))
    return normed_features, ranges, min_values


def dating_analyze(features, labels):
    """
    Analyze the dating data by visualization.

    :param features: the features of instances in dataset
    :param labels: the labels of instances in dataset
    :return: the visualized result
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(features[:, 1], features[:, 2],
               c=15. * np.array(labels), s=15. * np.array(labels))
    plt.xlabel('Percentage of Time Spent Playing Video Games')
    plt.ylabel('Liters of Ice Cream Consumed Weekly')
    plt.show()


def dating_test(ratio=0.10):
    """
    The function provides an example of using kNN to classify dating data.

    :param ratio: the ratio of test instances
    :return: total error rate on test instances
    """
    dating_features, dating_labels = txt2matrix()
    normed_features, ranges, min_values = auto_norm(dating_features)
    instances_num = normed_features.shape[0]
    test_num = int(instances_num * ratio)
    error_count = 0.0
    # the first test_num instances are chosen to test
    for i in range(test_num):
        classified = classify(normed_features[i, :], normed_features[test_num:, :],
                              dating_labels[test_num:], 3)
        print('The kNN classifier return: %d, the real answer is : %d' % (classified, dating_labels[i]))
        if classified != dating_labels[i]:
            error_count += 1.
    print('The total error rate is: %f%%' % (error_count / float(test_num) * 100))


def img2vec(filename='../../../dataset/digits/testDigits/0_0.txt'):
    """
    The function convert the digit stored in txt into vector (numpy array).

    :param filename: the file name (directory) where the digit stored
    :return: the numpy array
    """
    return_vec = np.zeros((1, 1024))
    file = open(filename)
    for i in range(32):
        line = file.readline()
        for j in range(32):
            return_vec[0, 32*i+j] = int(line[j])
    return return_vec


def handwriting_recognition():
    """
    The function provides an example of using kNN to classify handwriting digits.

    :return: total error rate on test instances
    """
    train_file_list = os.listdir('../../../dataset/digits/trainingDigits')
    train_num = len(train_file_list)
    train_features = np.zeros((train_num, 1024))
    train_labels = []
    for i in range(train_num):
        filename = train_file_list[i]
        prefix = filename.split('.')[0]
        label = int(prefix.split('_')[0])
        train_labels.append(label)
        train_features[i, :] = img2vec('../../../dataset/digits/trainingDigits/%s' % filename)

    test_file_list = os.listdir('../../../dataset/digits/testDigits')
    error_count = 0.0
    test_num = len(test_file_list)
    for i in range(test_num):
        filename = test_file_list[i]
        prefix = filename.split('.')[0]
        label = int(prefix.split('_')[0])
        test_vec = img2vec('../../../dataset/digits/testDigits/%s' % filename)
        classified = classify(test_vec, train_features, train_labels, 3)
        print('The kNN classifier return: %d, the real answer is : %d' % (classified, label))
        if classified != label:
            error_count += 1.
    print('The total error rate is: %f%%' % (error_count / float(test_num) * 100))


if __name__ == '__main__':
    handwriting_recognition()
