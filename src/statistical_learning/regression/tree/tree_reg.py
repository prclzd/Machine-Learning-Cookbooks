"""
This module implements Tree-based Regression by applying Classification And Regression Trees (CART) algorithm
without calling any pre-defined models and modules.
Two kinds of trees are created:
    (i) regression tree: split the data and create a binary tree with lowest error;
    (ii) model tree: model each leaf node as a piecewise linear model, which consists of multiple linear segments.

Author:
    Hailiang Zhao
"""
import numpy as np
import pprint


class TreeNode:
    def __init__(self, feature, value, right_branch, left_branch):
        """
        Initialization.

        :param feature: the chosen feature for the node
        :param value: the value of the chosen feature
        :param right_branch: the pointer to the right branch of the tree
        :param left_branch: the pointer to the left branch of the tree
        """
        self.feature = feature
        self.value = value
        self.right_branch = right_branch
        self.left_branch = left_branch


def load_dataset(filename="../../../../dataset/regression-examples/trees/ex0.txt"):
    # instances_arr contains instances with their data and labels
    # we do not split them into data_arr and label_arr
    n = len(open(filename).readline().split('\t'))
    instances_arr = []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        line_data = line.strip().split('\t')
        for i in range(n):
            line_arr.append(float(line_data[i]))
        instances_arr.append(line_arr)
    return instances_arr


def binary_split(data, feature, value):
    """
    According to the value of the chosen feature, split the dataset into two set.

    :param data:
    :param feature:
    :param value:
    :return:
    """
    highers = data[np.nonzero(data[:, feature] > value)[0], :]
    lowers = data[np.nonzero(data[:, feature] <= value)[0], :]
    return highers, lowers


def linear_solve(data):
    m, n = np.shape(data)
    X, y = np.mat(np.ones((m, n))), np.mat(np.ones((m, 1)))
    X[:, 1:n] = data[:, 0:n-1]
    y = data[:, -1]
    xTx = X.T * X
    if np.linalg.det(xTx) == 0.0:
        raise NameError('This matrix is singular, cannot do inverse! Try '
                        'increase the second value of ops.')
    ws = xTx.I * (X.T * y)
    return ws, X, y


def get_reg_leaf(data):
    """
    This function generates the value for a leaf node, by their mean value.

    :param data:
    :return:
    """
    return np.mean(data[:, -1])


def get_model_leaf(data):
    ws, _, _ = linear_solve(data)
    return ws


def get_reg_error(data):
    """
    This function returns the squared error of the target variables in a given dataset.

    :param data:
    :return:
    """
    return np.var(data[:, -1]) * np.shape(data)[0]


def get_model_error(data):
    ws, X, y = linear_solve(data)
    y_hat = X * ws
    return np.sum(np.power(y - y_hat, 2))


def choose_axis(data, leaf_type=get_reg_leaf, error_type=get_reg_error, ops=(1, 4)):
    """
    This function finds the best way to do a binary split on the data. If a “good” binary split can’t be found,
    then this function returns None and tells create_tree() to generate a leaf node. If a “good” split is found,
    the feature index (axis) and value of the split are returned.

    :param data: instances with their data and labels
    :param leaf_type: a reference to a function that we use to create the leaf node
    :param error_type: a reference to a function that will be used to calculate the squared deviation from the mean
    :param ops: a tuple of user-defined parameters to help with tree building
        --> err_reduce_tol: a tolerance on error reduction
        --> min_instances: the minimum number of instances to include in a split
    :return: the feature index (axis) and value of the split
    """
    # a tolerance on error reduction
    err_reduce_tol = ops[0]
    # the minimum number of instances to include in a split
    min_instances = ops[1]
    # exit if all values are equal, which means no feature to be chosen to split
    # thus we return a leaf node
    # (if this set is length 1, we don’t need to try to split the set and we can return)
    if len(set(data[:, -1].T.tolist()[0])) == 1:
        return None, leaf_type(data)

    m, n = np.shape(data)
    err = error_type(data)
    min_err = np.inf
    best_axis, best_value = 0, 0
    # we explore each feature and their corresponding values to find the best feature and value to split
    # the best one must provides the maximum error reduction
    for feature in range(n-1):
        for value in set(np.asarray(data[:, feature]).ravel()):
            highers, lowers = binary_split(data, feature, value)
            # if the separated two sets do not have enough instances, we choose not to split them under current choice
            if np.shape(highers)[0] < min_instances or np.shape(lowers)[0] < min_instances:
                continue
            new_err = error_type(highers) + error_type(lowers)
            if new_err < min_err:
                best_axis, best_value = feature, value
                min_err = new_err
    # if splitting the dataset improves the error by only a small amount, we choose not to split
    # and create a leaf node
    if (err - min_err) < err_reduce_tol:
        return None, leaf_type(data)

    highers, lowers = binary_split(data, best_axis, best_value)
    # if the separated two sets do not have enough instances, we choose not to split and create a leaf node
    if np.shape(highers)[0] < min_instances or np.shape(lowers)[0] < min_instances:
        return None, leaf_type(data)

    return best_axis, best_value


def create_tree(data, leaf_type=get_reg_leaf, error_type=get_reg_error, ops=(1, 4)):
    """
    Build a regression tree recursively.

    ===> The creation of the tree is significantly affected by ops, i.e., (1) the tolerance on error reduction;
    (2) the minimum number of instances to be included in a split. <===

    :param data:
    :param leaf_type:
    :param error_type:
    :param ops:
    :return:
    """
    feature, value = choose_axis(data, leaf_type, error_type, ops)
    if feature is None:
        return value
    tree = {'split_axis': feature, 'split_value': value}
    left, right = binary_split(data, feature, value)
    tree['right'] = create_tree(right, leaf_type, error_type, ops)
    tree['left'] = create_tree(left, leaf_type, error_type, ops)
    return tree


def is_tree(obj):
    return type(obj).__name__ == 'dict'


def get_mean(tree):
    """
    Collapse the tree into one node, and the single node's value is the average value of the two nodes.

    :param tree:
    :return:
    """
    if is_tree(tree['right']):
        tree['right'] = get_mean(tree['right'])
    if is_tree(tree['left']):
        tree['left'] = get_mean(tree['left'])
    return (tree['right'] + tree['left']) / 2.0


def evaluate_reg_tree(model, input_data):
    """
    Evaluate the value of y for the input_data with regression tree. In order to keep unification, it has the same
    parameters with evaluate_model_tree().

    :param model:
    :param input_data:
    :return:
    """
    return float(model)


def evaluate_model_tree(model, input_data):
    """
    Evaluate the value of y for the input_data with model tree. In order to keep unification, it has the same
    parameters with evaluate_reg_tree().

    :param model:
    :param input_data:
    :return:
    """
    n = np.shape(input_data)[1]
    X = np.mat(np.ones((1, n+1)))
    X[:, 1:n+1] = input_data
    return float(X * model)


def classify(tree, input_data, model_evaluation=evaluate_reg_tree):
    """
    Classify input_data (predict the value of y) with chosen tree model.

    :param tree:
    :param input_data:
    :param model_evaluation:
    :return:
    """
    if not is_tree(tree):
        return model_evaluation(tree, input_data)
    if input_data[tree['split_axis']] > tree['split_value']:
        if is_tree(tree['left']):
            return classify(tree['left'], input_data, model_evaluation)
        else:
            return model_evaluation(tree['left'], input_data)
    else:
        if is_tree(tree['right']):
            return classify(tree['right'], input_data, model_evaluation)
        else:
            return model_evaluation(tree['right'], input_data)


def forecast(tree, test_data, model_evaluation=evaluate_reg_tree):
    """
    Classify every instance in test_data and return the predicted values as a numpy matrix.

    :param tree:
    :param test_data:
    :param model_evaluation:
    :return:
    """
    m = len(test_data)
    y_hat = np.mat(np.zeros((m, 1)))
    for i in range(m):
        y_hat[i, 0] = classify(tree, np.mat(test_data[i]), model_evaluation)
    return y_hat


def prune(tree, test_data):
    """
    This function descend a inout tree until we reach a node with leaves. We test the leaves against
    test data and measure if merging the leaves would give us less error on the test set. If merging
    the nodes will reduce the error on the test set, we'll merge the nodes.
    The procedure is called 'post-pruning'.

    :param tree:
    :param test_data:
    :return:
    """
    # if the test data is empty, collapse the tree
    if np.shape(test_data)[0] == 0:
        return get_mean(tree)
    left, right = np.mat([]), np.mat([])
    if is_tree(tree['right']) or is_tree(tree['left']):
        left, right = binary_split(test_data, tree['split_axis'], tree['split_value'])
    if is_tree(tree['left']):
        tree['left'] = prune(tree['left'], left)
    if is_tree(tree['right']):
        tree['right'] = prune(tree['right'], right)
    if not is_tree(tree['left']) and not is_tree(tree['right']):
        left, right = binary_split(test_data, tree['split_axis'], tree['split_value'])
        err_before_merge = np.sum(np.power(left[:, -1] - tree['left'], 2)) + \
            np.sum(np.power(right[:, -1] - tree['right'], 2))
        tree_mean = (tree['left'] + tree['right']) / 2.0
        err_after_merge = np.sum(np.power(test_data[:, -1] - tree_mean, 2))
        if err_after_merge < err_before_merge:
            print('merging...')
            return tree_mean
        else:
            return tree
    return tree


if __name__ == '__main__':
    # 1: test binary_split()
    # test_mat = np.mat(np.eye(4))
    # mat0, mat1 = binary_split(test_mat, 1, 0.5)
    # print(mat0)
    # print(mat1)

    # 2: test create_tree()
    # data_mat = np.mat(load_dataset())
    # reg_tree = create_tree(data_mat, ops=(1, 4))
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(reg_tree)

    # 3: test post-pruning
    # data_mat2 = np.mat(load_dataset("../../../../dataset/regression-examples/trees/ex2.txt"))
    # reg_tree = create_tree(data_mat2, ops=(100, 4))
    # print('===> Original tree <===')
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(reg_tree)
    # print('\n===> Pruning... <===')
    #
    # test_data_mat = np.mat(load_dataset("../../../../dataset/regression-examples/trees/ex2test.txt"))
    # pruned_tree = prune(reg_tree, test_data_mat)
    # print('\n===> Pruned tree <===')
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(pruned_tree)

    # 4: test model tree
    # data_mat3 = np.mat(load_dataset("../../../../dataset/regression-examples/trees/exp2.txt"))
    # model_tree = create_tree(data_mat3, get_model_leaf, get_model_error, (1, 10))
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(model_tree)

    # 5: evaluation
    train_mat = np.mat(load_dataset("../../../../dataset/regression-examples/trees/bikeSpeedVsIq_train.txt"))
    test_mat = np.mat(load_dataset("../../../../dataset/regression-examples/trees/bikeSpeedVsIq_test.txt"))

    bike_reg_tree = create_tree(train_mat, ops=(1, 20))
    print('===> Regression Tree <===')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bike_reg_tree)
    y_hat_reg = forecast(bike_reg_tree, test_mat[:, 0])
    print('\nThe R^2 value is: %f\n\n' % np.corrcoef(y_hat_reg, test_mat[:, 1], rowvar=False)[0, 1])

    bike_model_tree = create_tree(train_mat, get_model_leaf, get_model_error, ops=(1, 20))
    print('===> Regression Tree <===')
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bike_model_tree)
    y_hat_model = forecast(bike_model_tree, test_mat[:, 0], evaluate_model_tree)
    print('\nThe R^2 value is: %f' % np.corrcoef(y_hat_model, test_mat[:, 1], rowvar=False)[0, 1])
