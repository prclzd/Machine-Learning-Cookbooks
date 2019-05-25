"""
This module provides the standard and related implementation of Linear Regression, including Standard regression
function and corresponding shrinking strategies (Ridge regression, the LASSO, Forward stagewise regression).

Author:
    Hailiang Zhao
"""
import numpy as np
import matplotlib.pyplot as plt


def load_dataset(filename='../../../../dataset/regression-examples/ex0.txt'):
    features_num = len(open(filename).readline().split('\t')) - 1
    data_arr, label_arr = [], []
    fr = open(filename)
    for line in fr.readlines():
        line_arr = []
        line_data = line.strip().split('\t')
        for i in range(features_num):
            line_arr.append(float(line_data[i]))
        data_arr.append(line_arr)
        label_arr.append(float(line_data[-1]))
    return data_arr, label_arr


def standard_regress(data_arr, label_arr):
    """
    This function implements the standard linear regression, and returns the calculated
    solution.

    :param data_arr:
    :param label_arr:
    :return:
    """
    X, y = np.mat(data_arr), np.mat(label_arr).T
    xTx = X.T * X
    # judge whether the inverse of X^T * X exists
    if np.linalg.det(xTx) == 0.:
        print('This matrix (X^T * X) is singular, can not do inverse!')
        return
    # what we returned is the calculated best weights under Ordinary Least Squares (OLS)
    ws = xTx.I * (X.T * y)
    return ws


def std_reg_test(data_arr, label_arr):
    """
    This function calculated the predicted labels of data_arr by calling locally weighted
    linear regression, and plots the figure.

    :param data_arr:
    :param label_arr:
    :return:
    """
    we = standard_regress(data_arr, label_arr)
    X = np.mat(data_arr)
    y = np.mat(label_arr)

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    ax.scatter(X[:, 1].flatten().A[0], y.T[:, 0].flatten().A[0])

    X_copy = X.copy()
    X_copy.sort(0)
    y_predicted = X_copy * we
    ax.plot(X_copy[:, 1], y_predicted)
    plt.show()

    return y_predicted


def locally_weighted_regress(test_data, data_arr, label_arr, k=1.0):
    """
    This function implements the locally weighted linear regression, and returns the
    predicted value (label) of test_data.

    :param test_data:
    :param data_arr:
    :param label_arr:
    :param k: controls how quickly the decay happens
        WIth k=1.0, he weights are so large that they appear to weight all the data equally, and you have the
        same best-fit line as using standard regression.
    :return:
    """
    X, y = np.mat(data_arr), np.mat(label_arr).T
    m = np.shape(X)[0]
    # use a kernel function to weight nearby points more heavily than other points
    # the 'distance' is from test_data (an instance) to data-arr (size m)
    weights = np.mat(np.eye(m))
    for j in range(m):
        diff_mat = test_data - X[j, :]
        weights[j, j] = np.exp(diff_mat * diff_mat.T / (-2.*k**2))
    xTx = X.T * (weights * X)
    if np.linalg.det(xTx) == 0.:
        print('This matrix (X^T * X) is singular, can not do inverse!')
        return
    ws = xTx.I * (X.T * (weights * y))
    return test_data * ws


def lw_reg_test(test_arr, data_arr, label_arr, k=1.0):
    """
    This function calculated the predicted labels of test_arr by calling locally weighted
    linear regression, and plots the figure.

    If the data instances have multiple dimensions (>=2), we only plot 'the value second dimension' vs.
    'the value of label'.

    :param test_arr:
    :param data_arr:
    :param label_arr:
    :param k: controls how quickly the decay happens
    :return:
    """
    m = np.shape(test_arr)[0]
    y_predicted = np.zeros(m)
    # locally weighted linear regression uses the whole dataset each time a calculation
    # is needed, similar to kNN
    for i in range(m):
        y_predicted[i] = locally_weighted_regress(test_arr[i], data_arr, label_arr, k)

    # plot figure
    X = np.mat(data_arr)
    sorted_idx = X[:, 1].argsort(0)
    X_sort = X[sorted_idx][:, 0, :]

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    ax.plot(X_sort[:, 1], y_predicted[sorted_idx])
    ax.scatter(X[:, 1].flatten().A[0], np.mat(label_arr).T.flatten().A[0], s=2, c='red')
    plt.show()

    return y_predicted


def ridge_regress(X, y, lam=0.2):
    """
    This function implements the ridge linear regression, and returns the calculated
    solution.

    When we have more data points than features, the matrix xTx is surely not fully-ranked,
    thus it is non-singular. As a result, xTx does not have inverse. We add a user-defined
    scalar value lam (lambda) to construct an identity matrix to make xTx inversable.
    We can use the lambda value to impose a maximum value on the sum of all our ws.
    By imposing this penalty, we can decrease unimportant parameters. This decreasing i
    s known as shrinkage in statistics.

    The equation for ridge regression is the same as our regular least-squares regression and
    imposing the following constraint: \sum_{k=1}^n w_k^2 \leq \lambda.

    Thus, Ridge Regression is the implementation of regularization.

    :param X:
    :param y:
    :param lam:
    :return:
    """
    xTx = X.T * X
    regularized_xTx = xTx + np.eye(np.shape(X)[1]) * lam
    if np.linalg.det(regularized_xTx) == 0.:
        print('The normalized matrix is singular, can not do inverse!')
        return
    ws = regularized_xTx.I * (X.T * y)
    return ws


def compare_ridges(data_arr, label_arr):
    """
    This function compares 30 different lambdas in Ridge Regression  adn returns the calculated
    solutions in ws_mat.

    :param data_arr:
    :param label_arr:
    :return:
    """
    X, y = np.mat(data_arr), np.mat(label_arr).T

    # normalization
    # to use ridge regression and all shrinkage methods, we need to first normalize your features
    # we normalized our data to give each feature equal importance regardless of the units it was measured in
    y_mean = np.mean(y, 0)
    y = y - y_mean
    X_means = np.mean(X, 0)
    X_var = np.var(X, 0)
    X = (X - X_means) / X_var

    lambda_num = 30
    ws_mat = np.zeros((lambda_num, np.shape(X)[1]))
    for i in range(lambda_num):
        ws = ridge_regress(X, y, np.exp(i-10))
        ws_mat[i, :] = ws.T

    # plot figure on coefficients vs. lambdas
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    ax.plot(ws_mat)
    plt.show()

    return ws_mat


def ridge_reg_test(data_arr, label_arr, lam=0.2):
    """
        This function calculated the predicted labels of data_arr by calling ridge
        linear regression, and plots the figure.

        :param data_arr:
        :param label_arr:
        :param lam:
        :return:
        """
    X, y = np.mat(data_arr), np.mat(label_arr).T
    we = ridge_regress(X, y, lam)
    y_predicted = X * we

    return y_predicted


def forward_stagewise_regress(data_arr, label_arr, eps=0.01, iterations=100):
    """
    This function implements the Forward Stagewise Regression, and returns the calculated solutions in
    different iterations.

    The equation for LASSO (Least Absolute Shrinkage and Selection Operator) is the same as our regular
    least-squares regression and imposing the following constraint: \sum_{k=1}^n |w_k| \leq \lambda. The
    mathematical difference of the constraints may seem trivial, but it makes things a lot harder to solve.
    To solve this we now need a quadratic programming algorithm. Instead of using the quadratic solver,
    Forward Stagewise Regression is adopted, which is an easier algorithm to implements LASSO.

    :param data_arr:
    :param label_arr:
    :param eps: the step size to take at each iteration
    :param iterations: the number of iterations
    :return:
    """
    pass


def forward_stagewise_reg_test(data_arr, label_arr, eps=0.01, iterations=100):
    pass


def abalone_test():
    data_arr, label_arr = load_dataset('../../../../dataset/regression-examples/abalone.txt')
    print('===> Test Standard Regression <===')
    we = standard_regress(data_arr[0:99], label_arr[0:99])
    y_predicted = np.mat(data_arr[0:99]) * we
    print('Residual Sum of Squares (RSS) error on training set is: %f' %
          (residual_sum_squares(label_arr[0:99], y_predicted.T.A)))
    y_predicted = np.mat(data_arr[100:199]) * we
    print('Residual Sum of Squares (RSS) error on test set is: %f' %
          (residual_sum_squares(label_arr[100:199], y_predicted.T.A)))

    print('===> Test Locally Weighted Linear Regression <===')
    for k in [0.1, 1, 10]:
        print('k = %f' % k)
        y_predicted = lw_reg_test(data_arr[0:99], data_arr[0:99], label_arr[0:99], k)
        print('Residual Sum of Squares (RSS) error on training set is: %f' %
              (residual_sum_squares(label_arr[0:99], y_predicted.T)))
        # work on new data
        y_predicted = lw_reg_test(data_arr[100:199], data_arr[0:99], label_arr[0:99], k)
        print('Residual Sum of Squares (RSS) error on test set is: %f\n' %
              (residual_sum_squares(label_arr[100:199], y_predicted.T)))

    print('===> Test Ridge Linear Regression <===')
    for lam in [np.exp(-5), 0.02, np.exp(5)]:
        print('lambda = %f' % lam)
        we = ridge_regress(np.mat(data_arr[0:99]), np.mat(label_arr[0:99]).T, lam)
        y_predicted = np.mat(data_arr[0:99]) * we
        print('Residual Sum of Squares (RSS) error on training set is: %f' %
              (residual_sum_squares(label_arr[0:99], y_predicted.T.A)))
        y_predicted = np.mat(data_arr[100:199]) * we
        print('Residual Sum of Squares (RSS) error on test set is: %f' %
              (residual_sum_squares(label_arr[100:199], y_predicted.T.A)))

    print('===> Test Forward Stagewise Regression <===')
    pass


def get_correlation(y_predicted, y_actual):
    """
    Calculate the correlation coefficients between y_estimate and y_actual.

    :param y_predicted:
    :param y_actual:
    :return:
    """
    return np.corrcoef(y_predicted.T, y_actual)


def residual_sum_squares(y, y_predicted):
    return ((y - y_predicted)**2).sum()


if __name__ == '__main__':
    # use example data to test standard regression and locally weighted linear regression
    # data, labels = load_dataset()
    # std_reg_test(data, labels)
    # lw_reg_test(data, data, labels, k=0.03)

    # use abalone data to test ridge linear regression
    # data, labels = load_dataset('../../../../dataset/regression-examples/abalone.txt')
    # compare_ridges(data, labels)
    # ridge_reg_test(data, labels, lam=0.02)

    abalone_test()
