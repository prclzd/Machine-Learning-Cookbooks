"""
This module defines several functions and a class about Support Vector Machine (SVM) algorithms
without call any pre-defined models.

Author:
    Hailiang Zhao
"""
import random
import numpy as np


def load_dataset(filename='../../../dataset/svm-examples/testSet.txt'):
    features, labels = [], []
    fr = open(filename)
    for line in fr.readlines():
        line_data = line.strip().split()
        features.append([float(line_data[0]), float(line_data[1])])
        labels.append(float(line_data[2]))
    return features, labels


def random_select(i, m):
    """
    Find an index j from [0, m] which is noe equal to i.

    :param i: the index which can not be equal to (the index of the first alpha)
    :param m: the upper bound of the range (the total number of alphas)
    :return: the found index j (the index of the second alpha)
    """
    j = i
    while j == i:
        j = int(random.uniform(0, m))
    return j


def clip(alpha, upper_bound, lower_bound):
    """
    Clip alpha into [lower_bound, upper_bound].

    :param alpha:
    :param upper_bound:
    :param lower_bound:
    :return:
    """
    if alpha > upper_bound:
        alpha = upper_bound
    if alpha < lower_bound:
        alpha = lower_bound
    return alpha


def simplified_smo(features, labels, c, precision, iterations):
    """
    This is the implementation of simplified Platt's Sequential Minimal Optimization (SMO) algorithm.

    :param features:
    :param labels:
    :param c:
    :param precision:
    :param iterations:
    :return:
    """
    features_matrix = np.mat(features)
    labels_matrix = np.mat(labels).transpose()
    m, n = np.shape(features_matrix)

    alphas = np.mat(np.zeros((m, 1)))
    b = 0.

    # it holds a count of the number of times that we go through the dataset
    # without any alphas changing
    it = 0
    while it < iterations:
        # the variable alpha_pairs_changed is used to record if the attempt to
        # optimize any alphas worked
        alpha_pairs_changed = 0
        for i in range(m):
            predicted_i = float(np.multiply(alphas, labels_matrix).T *
                                (features_matrix * features_matrix[i, :].T)) + b
            error_i = predicted_i - float(labels_matrix[i])
            # enter optimization procedure if alphas can be optimized
            # it's not worth trying to optimize these alphas who are bound
            # and can not be increased or decreased
            if (labels_matrix[i] * error_i < -precision) and (alphas[i] < c) or \
                    (labels_matrix[i] * error_i > precision) and (alphas[i] > 0):
                j = random_select(i, m)
                predicted_j = float(np.multiply(alphas, labels_matrix).T *
                                    (features_matrix * features_matrix[j, :].T)) + b
                error_j = predicted_j - float(labels_matrix[j])
                alpha_i_old, alpha_j_old = alphas[i].copy(), alphas[j].copy()
                # guarantee alphas stay between 0 and c (the constraint)
                if labels_matrix[i] != labels_matrix[j]:
                    lower_bound = max(0, alphas[j] - alphas[i])
                    upper_bound = min(c, c + alphas[j] - alphas[i])
                else:
                    lower_bound = max(0, alphas[j] + alphas[i] - c)
                    upper_bound = min(c, alphas[j] + alphas[i])
                if lower_bound == upper_bound:
                    print('lower_bound = upper_bound')
                    continue

                eta = 2.0 * features_matrix[i, :] * features_matrix[j, :].T - \
                    features_matrix[i, :] * features_matrix[i, :].T - \
                    features_matrix[j, :] * features_matrix[j, :].T
                # if eta is 0, there’s a messy way to calculate the new alpha[j]
                # thus, we just quit the current iteration in the simplified version
                if eta >= 0:
                    print('eta >= 0')
                    continue

                # update alphas[i] and alphas[j]
                alphas[j] = alphas[j] - labels_matrix[j] * (error_i - error_j) / eta
                alphas[j] = clip(alphas[j], upper_bound, lower_bound)
                if abs(alphas[j] - alpha_j_old) < 0.00001:
                    print('alpha_j is not moving enough!')
                    continue
                # change alphas[i] by the same amount with an opposite direction
                alphas[i] = alphas[i] + labels_matrix[j] * labels_matrix[i] * (alpha_j_old - alphas[j])

                # update b
                b1 = b - error_i - \
                     labels_matrix[i] * (alphas[i] - alpha_i_old) * features_matrix[i, :] * features_matrix[i, :].T - \
                     labels_matrix[j] * (alphas[j] - alpha_j_old) * features_matrix[i, :] * features_matrix[j, :].T
                b2 = b - error_j - \
                     labels_matrix[i] * (alphas[i] - alpha_i_old) * features_matrix[i, :] * features_matrix[j, :].T - \
                     labels_matrix[j] * (alphas[j] - alpha_j_old) * features_matrix[j, :] * features_matrix[j, :].T
                if 0 < alphas[i] < c:
                    b = b1
                elif 0 < alphas[j] < c:
                    b = b2
                else:
                    b = (b1 + b2) / 2.

                alpha_pairs_changed += 1
                print('Iterations: %d, i: %d, pairs changed %d' % (it, i, alpha_pairs_changed))
        if alpha_pairs_changed == 0:
            it += 1
        else:
            it = 0
        print('Iteration number: %d' % it)
    return b, alphas


class SmoOptimization:
    """
    The class defines the standard implementation of Platt's SMO algorithm for SVMs.

    """
    def __init__(self, X, y, c, precision, iterations, use_kernel=False, kernel_tuple=('lin', 0)):
        """
        Initialization.

        :param X: the np.mat of dataset features
        :param y: the np.mat of dataset labels
        :param c: c controls the balance between making sure all of the examples have a margin of at least 1.0
            and making the margin as wide as possible. If c is large, the classifier will try to make all of
            the examples properly classified by the separating hyperplane.
        :param precision: the precision need-to-be realized
        :param iterations:
        :param use_kernel:
        :param kernel_tuple: indicate that which kind of kernel is used.
            The former is the name of chosen kernel, the latter os the parameter that determines the 'reach' (how
            quickly this falls off to zero).
            If the kernel is linear, the parameter has no use, we simply set it as zero;
            if the kernel is Radial Bias Function (RBF), the parameter os sigma.
        """
        # there has no need to use private variables
        self.X = X
        self.y = y
        self.c = c
        self.precision = precision
        self.iterations = iterations

        self.m = np.shape(X)[0]
        self.alphas = np.mat(np.zeros((self.m, 1)))
        self.b = 0
        # the first column is a flag bit stating whether the eCache is valid,
        # and the second column is the actual error value
        self.errors_cache = np.mat(np.zeros((self.m, 2)))

        self.use_kernel = use_kernel
        # set newly mapped 'distances' (or features) by the chosen kernel types
        # (default is 'lin' for linear-separable dataset)
        self.mapped_X = np.mat(np.zeros((self.m, self.m)))
        for i in range(self.m):
            self.mapped_X[:, i] = SmoOptimization.map_with_kernels(self.X, self.X[i, :], kernel_tuple)

    def get_error(self, k):
        """
        Get the classification error of the k-th instance.

        :param k:
        :return:
        """
        if self.use_kernel:
            predicted = float(np.multiply(self.alphas, self.y).T * self.mapped_X[:, k] + self.b)
        else:
            predicted = float(np.multiply(self.alphas, self.y).T * (self.X * self.X[k, :].T)) + self.b
        return predicted - float(self.y[k])

    def select_j(self, i, error_i):
        """
        According to the value of i and error_i, select the best j, which should have the
        maximum step size.

        :param i:
        :param error_i:
        :return:
        """
        max_k, max_delta_error, error_j = -1, 0, 0
        self.errors_cache[i] = [1, error_i]
        # valid means that it has been calculated
        valid_cache_list = np.nonzero(self.errors_cache[:, 0].A)[0]
        if len(valid_cache_list) > 1:
            for k in valid_cache_list:
                if k == i:
                    continue
                error_k = self.get_error(k)
                delta_error = abs(error_i - error_k)
                if delta_error > max_delta_error:
                    max_k = k
                    max_delta_error = delta_error
                    error_j = error_k
            return max_k, error_j
        # for the first through the loop, we just randomly select an alpha
        # actually there exist more sophisticated ways of handling the first-time case
        else:
            j = random_select(i, self.m)
            error_j = self.get_error(j)
        return j, error_j

    def update_error_cache(self, k):
        error_k = self.get_error(k)
        self.errors_cache[k] = [1, error_k]

    def inner_loop(self, i):
        """
        This function defines the procedure about what we do when we has already get
        the first (outer) alpha.

        :param i:
        :return:
        """
        error_i = self.get_error(i)
        if (self.y[i] * error_i < -self.precision) and (self.alphas[i] < self.c) or \
                (self.y[i] * error_i > self.precision) and (self.alphas[i] > 0):
            j, error_j = self.select_j(i, error_i)
            alpha_i_old, alpha_j_old = self.alphas[i].copy(), self.alphas[j].copy()
            # guarantee alphas stay between 0 and c (the constraint)
            if self.y[i] != self.y[j]:
                lower_bound = max(0, self.alphas[j] - self.alphas[i])
                upper_bound = min(self.c, self.c + self.alphas[j] - self.alphas[i])
            else:
                lower_bound = max(0, self.alphas[j] + self.alphas[i] - self.c)
                upper_bound = min(self.c, self.alphas[j] + self.alphas[i])
            if lower_bound == upper_bound:
                print('lower_bound = upper_bound')
                return 0

            if self.use_kernel:
                eta = 2.0 * self.mapped_X[i, j] - self.mapped_X[i, i] - self.mapped_X[j, j]
            else:
                eta = 2.0 * self.X[i, :] * self.X[j, :].T - \
                      self.X[i, :] * self.X[i, :].T - \
                      self.X[j, :] * self.X[j, :].T
            # if eta is 0, there’s a messy way to calculate the new alpha[j]
            # thus, we just quit
            if eta >= 0:
                print('eta >= 0')
                return 0

            # update self.alphas[i] and self.alphas[j], and then update the error cache
            self.alphas[j] -= self.y[j] * (error_i - error_j) / eta
            self.alphas[j] = clip(self.alphas[j], upper_bound, lower_bound)
            self.update_error_cache(j)
            if abs(self.alphas[j] - alpha_j_old) < 0.00001:
                print('alpha_j is not moving enough!')
                return 0
            self.alphas[i] += self.y[j] * self.y[i] * (alpha_j_old - self.alphas[j])
            self.update_error_cache(i)

            # update b
            if self.use_kernel:
                b1 = self.b - error_i - self.y[i] * (self.alphas[i] - alpha_i_old) * self.mapped_X[i, i] - \
                    self.y[j] * (self.alphas[j] - alpha_j_old) * self.mapped_X[i, j]
                b2 = self.b - error_j - self.y[i] * (self.alphas[i] - alpha_i_old) * self.mapped_X[i, j] - \
                    self.y[j] * (self.alphas[j] - alpha_j_old) * self.mapped_X[j, j]
            else:
                b1 = self.b - error_i - \
                    self.y[i] * (self.alphas[i] - alpha_i_old) * self.X[i, :] * self.X[i, :].T - \
                    self.y[j] * (self.alphas[j] - alpha_j_old) * self.X[i, :] * self.X[j, :].T
                b2 = self.b - error_j - \
                    self.y[i] * (self.alphas[i] - alpha_i_old) * self.X[i, :] * self.X[j, :].T - \
                    self.y[j] * (self.alphas[j] - alpha_j_old) * self.X[j, :] * self.X[j, :].T
            if 0 < self.alphas[i] < self.c:
                self.b = b1
            elif 0 < self.alphas[j] < self.c:
                self.b = b2
            else:
                self.b = (b1 + b2) / 2.
            return 1
        return 0

    @staticmethod
    def smo(features, labels, c, precision, iterations, use_kernel=False, kernel_tuple=('lin', 0)):
        opt = SmoOptimization(np.mat(features), np.mat(labels).transpose(), c, precision,
                              iterations, use_kernel, kernel_tuple)
        it, entire_set = 0, True
        alpha_pair_changed = 0

        while it < iterations and (alpha_pair_changed > 0 or entire_set):
            alpha_pair_changed = 0
            if entire_set:
                for i in range(opt.m):
                    alpha_pair_changed += opt.inner_loop(i)
                    print('Full set, iter: %d, i: %d, pairs changed %d times' % (it, i, alpha_pair_changed))
                it += 1
            else:
                non_bounded_list = np.nonzero((opt.alphas.A > 0) * (opt.alphas.A < opt.c))[0]
                for i in non_bounded_list:
                    alpha_pair_changed += opt.inner_loop(i)
                    print('Non-bounded, iter: %d, i: %d, pairs changed %d' % (it, i, alpha_pair_changed))
                it += 1
            if entire_set:
                entire_set = False
            elif alpha_pair_changed == 0:
                entire_set = True
            print('Iteration number: %d' % it)
        return opt.b, opt.alphas

    @staticmethod
    def classify(feature, alphas, b, features, labels):
        X = np.mat(features)
        y = np.mat(labels).transpose()
        m, n = np.shape(X)
        w = np.zeros((n, 1))
        for i in range(m):
            w += np.multiply(alphas[i] * y[i], X[i, :].T)
        predicted = np.mat(feature) * np.mat(w) + b
        return np.sign(predicted)

    @staticmethod
    def map_with_kernels(X, compared_vec, kernel_tuple):
        """
        Calculate the new features (distances) of the compared vector (instance) by the
        chosen kernel function. The new features is of size m (originally it's n).

        :param X:
        :param compared_vec:
        :param kernel_tuple:
        :return:
        """
        m, n = np.shape(X)
        distances = np.mat(np.zeros((m, 1)))
        # in the case of the linear kernel, a dot product is taken between the two inputs,
        # which are the full dataset and a row of the dataset
        if kernel_tuple[0] == 'lin':
            distances = X * compared_vec.T
        elif kernel_tuple[0] == 'rbf':
            for i in range(m):
                delta = X[i, :] - compared_vec
                distances[i] = delta * delta.T
            distances = np.exp(distances / (-1 * kernel_tuple[1] ** 2))
        # add more kernels later
        else:
            raise NameError('The kernel can not be recognized!')
        return distances


def smo_test():
    data_arr, label_arr = load_dataset()
    lin_b, lin_alphas = SmoOptimization.smo(data_arr, label_arr, 0.6, 0.001, 40)
    error_count = 0.
    for i in range(len(data_arr)):
        if SmoOptimization.classify(data_arr[i], lin_alphas, lin_b, data_arr, label_arr) != np.sign(label_arr[i]):
            error_count += 1
    print('The training error rate is: %f%%' % (error_count * 100 / len(data_arr)))


def rbf_kernel_test(sigma=1.3):
    data_arr, label_arr = load_dataset('../../../dataset/svm-examples/testSetRBF.txt')
    rbf_b, rbf_alphas = SmoOptimization.smo(
        data_arr, label_arr, 200, 0.0001, 10000, use_kernel=True, kernel_tuple=('rbf', sigma)
    )
    data_mat, label_mat = np.mat(data_arr), np.mat(label_arr).transpose()
    sv_idx = np.nonzero(rbf_alphas.A > 0)[0]
    support_vectors = data_mat[sv_idx]
    support_labels = label_mat[sv_idx]
    print('There are %d support vectors' % len(sv_idx))
    print('The supported vectors are:\n', support_vectors, '\nthere labels are:\n', support_labels)

    m, n = np.shape(data_mat)
    error_count = 0.
    for i in range(m):
        sv_mapped_features = SmoOptimization.map_with_kernels(support_vectors, data_mat[i, :], ('rbf', sigma))
        predicted = sv_mapped_features.T * np.multiply(support_labels, rbf_alphas[sv_idx]) + rbf_b
        if np.sign(label_arr[i]) != np.sign(predicted):
            error_count += 1
    print('The training error rate is: %f%%' % (error_count * 100 / m))

    data_arr, label_arr = load_dataset('../../../dataset/svm-examples/testSetRBF2.txt')
    data_mat, label_mat = np.mat(data_arr), np.mat(label_arr).transpose()
    m, n = np.shape(data_mat)
    error_count = 0.
    for i in range(m):
        sv_mapped_features = SmoOptimization.map_with_kernels(support_vectors, data_mat[i, :], ('rbf', sigma))
        predicted = sv_mapped_features.T * np.multiply(support_labels, rbf_alphas[sv_idx]) + rbf_b
        if np.sign(label_arr[i]) != np.sign(predicted):
            error_count += 1
    print('The test error rate is: %f%%' % (error_count * 100 / m))


if __name__ == '__main__':
    # smo_test()
    rbf_kernel_test()
