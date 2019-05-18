"""
This module defines the Decision Tree classification (using ID3 algorithm) without call any pre-defined models.

Author:
    Hailiang Zhao
"""
import math
import operator
import matplotlib.pyplot as plt
import pickle


def create_dataset():
    """
    The dataset contains five animals pulled from the sea and asks if they can survive
    without coming to the surface and if they have flippers. We finally decide whether
    they are fish or not.

    :return: the created dataset and axis names
    """
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    axis_names = ['no surfacing', 'flippers']
    return dataset, axis_names


def calculate_shannon_entropy(dataset):
    """
    This function calculates the Shannon entropy of a dataset (recognize the probability
    of each instance and math.log them).

    :param dataset: the features with labels of the dataset
    :return: the shannon entropy of the dataset
    """
    instances_num = len(dataset)
    labels_count = {}
    for instance in dataset:
        label = instance[-1]
        if label not in labels_count.keys():
            labels_count[label] = 0
        labels_count[label] += 1
    # the higher the entropy, the more mixed up the data is
    shannon_entropy = 0.0
    for key in labels_count:
        prob = float(labels_count[key]) / instances_num
        shannon_entropy -= prob * math.log(prob, 2)
    return shannon_entropy


def split_dataset(dataset, axis, value):
    """
    Split the dataset if the value of the chosen axis is right.

    :param dataset: the features with labels of the dataset
    :param axis: the chosen axis
    :param value: the desired value of the chosen axis
    :return: the sub-dataset with the chosen axis erased
    """
    sub_dataset = []
    for instance in dataset:
        if instance[axis] == value:
            reduced_instance = instance[:axis]
            reduced_instance.extend(instance[axis+1:])
            sub_dataset.append(reduced_instance)
    return sub_dataset


def choose_axis(dataset):
    """
    This function choose the best axis to split for current dataset.

    :param dataset: the features with labels of the dataset
    :return: the best axis who should be chosen to split the dataset
    """
    features_num = len(dataset[0]) - 1
    base_entropy = calculate_shannon_entropy(dataset)
    best_info_gain = 0.0
    best_axis = -1
    for i in range(features_num):
        features_list = [instance[i] for instance in dataset]
        unique_values = set(features_list)
        new_entropy = 0.0
        for value in unique_values:
            sub_features = split_dataset(dataset, i, value)
            prob = len(sub_features) / float(len(dataset))
            new_entropy += prob * calculate_shannon_entropy(sub_features)
        info_gain = base_entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_axis = i
    return best_axis


def majority_count(labels_list):
    """
    This function returns the majority label in a sub-dataset (when dataset has run out of attributes).

    :param labels_list: the list where labels of sub-dataset stored
    :return: the majority label
    """
    labels_count = {}
    for vote in labels_list:
        if vote not in labels_count.keys():
            labels_count[vote] = 0
        labels_count[vote] += 1
    sorted_labels_count = sorted(labels_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_labels_count[0][0]


def create_tree(dataset, axis_names):
    """
    The function creates the decision tree.
    Two exit conditions:
        (1) the dataset has run out of attributes but the class labels are not the same (then majority count;
        (2) all the instances in the branch are the same class.

    :param dataset: the features with labels of the dataset
    :param axis_names: the list of names of attributes (axis)
    :return: the created tree, stored in dict
    """
    labels_list = [instance[-1] for instance in dataset]
    # all the instances in the branch are the same class, return the class
    if labels_list.count(labels_list[0]) == len(labels_list):
        return labels_list[0]
    # attributes have been run out of
    if len(dataset[0]) == 1:
        return majority_count(labels_list)
    best_axis = choose_axis(dataset)
    best_axis_name = axis_names[best_axis]
    tree = {best_axis_name: {}}
    del(axis_names[best_axis])
    feature_values = [instance[best_axis] for instance in dataset]
    unique_values = set(feature_values)
    for value in unique_values:
        sub_axis_names = axis_names[:]
        tree[best_axis_name][value] = create_tree(split_dataset(dataset, best_axis, value), sub_axis_names)
    return tree


# global settings on node types and arrow type (store into dict, as parameters)
decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(node_txt, center_point, parent_point, node_type):
    """
    Store information of nodes into global variable create_plot.ax1.

    :param node_txt:
    :param center_point:
    :param parent_point:
    :param node_type:
    :return:
    """
    create_plot.ax1.annotate(
        node_txt, xy=parent_point, xycoords='axes fraction',
        xytext=center_point, textcoords='axes fraction',
        va='center', ha='center', bbox=node_type, arrowprops=arrow_args
    )


def add_line_txt(center_point, parent_point, txt):
    x_middle = (parent_point[0] - center_point[0]) / 2.0 + center_point[0]
    y_middle = (parent_point[1] - center_point[1]) / 2.0 + center_point[1]
    create_plot.ax1.text(x_middle, y_middle, txt)


def plot_tree(tree, parent_point, node_txt):
    leafs_num = get_leafs_num(tree)
    first_key = list(tree.keys())[0]
    center_point = (plot_tree.xOff + (1.0 + float(leafs_num)) / 2.0 / plot_tree.totalW, plot_tree.yOff)
    add_line_txt(center_point, parent_point, node_txt)
    plot_node(first_key, center_point, parent_point, decision_node)
    second_dict = tree[first_key]
    plot_tree.yOff = plot_tree.yOff - 1.0 / plot_tree.totalD
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            plot_tree(second_dict[key], center_point, str(key))
        else:
            plot_tree.xOff = plot_tree.xOff + 1.0 / plot_tree.totalW
            add_line_txt((plot_tree.xOff, plot_tree.yOff), center_point, str(key))
            plot_node(second_dict[key], (plot_tree.xOff, plot_tree.yOff), center_point, leaf_node)
    plot_tree.yOff = plot_tree.yOff + 1.0 / plot_tree.totalD


def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    create_plot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # get the width and height (depth) of the tree and store them
    # into plot_tree.totalW, plot_tree.totalD, respectively
    plot_tree.totalW = float(get_leafs_num(tree))
    plot_tree.totalD = float(get_tree_depth(tree))
    # get the position where the next node stored
    plot_tree.xOff = -0.5 / plot_tree.totalW
    plot_tree.yOff = 1.0
    plot_tree(tree, (0.5, 1.0), '')
    plt.show()


def get_leafs_num(tree):
    """
    Recursively get the number of leaf nodes in the created tree.

    :param tree: a dict where the tree stored
    :return: the number of leaf nodes
    """
    leafs_num = 0
    first_key = list(tree.keys())[0]
    second_dict = tree[first_key]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            leafs_num += get_leafs_num(second_dict[key])
        else:
            leafs_num += 1
    return leafs_num


def get_tree_depth(tree):
    """
    Recursively get the depth of the created tree.

    :param tree:
    :return:
    """
    depth = 0
    first_key = list(tree.keys())[0]
    second_dict = tree[first_key]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            current_depth = 1 + get_tree_depth(second_dict[key])
        else:
            current_depth = 1
        if current_depth > depth:
            depth = current_depth
    return depth


def classify(test_vec, tree, axis_names):
    first_key = list(tree.keys())[0]
    second_dict = tree[first_key]
    feature_index = axis_names.index(first_key)
    for key in second_dict.keys():
        if test_vec[feature_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                classified = classify(test_vec, second_dict[key], axis_names)
            else:
                classified = second_dict[key]
    return classified


def store_tree(tree, filename):
    fw = open(filename, 'wb')
    pickle.dump(tree, fw)
    fw.close()


def grab_tree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


def lenses(filename='../../../dataset/lenses.txt'):
    fr = open(filename)
    lenses_dataset = [instance.strip().split('\t') for instance in fr.readlines()]
    lenses_axis_names = ['age', 'prescript', 'astigmatic', 'tearRate']
    tree = create_tree(lenses_dataset, lenses_axis_names)
    create_plot(tree)


if __name__ == '__main__':
    my_dataset, my_axis_names = create_dataset()
    # the axis_names variable is destroyed by create_tree() function, thus we need a copy
    # original_axis_names = my_axis_names.copy()
    # my_tree = create_tree(my_dataset, my_axis_names)
    # store_tree(my_tree, 'fish_or_not')
    # print(grab_tree('fish_or_not'))
    lenses()
