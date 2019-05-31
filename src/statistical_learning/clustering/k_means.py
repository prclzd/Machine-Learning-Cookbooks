"""
This module contains functions on some unsupervised learning algorithms, especially clustering.
"""
import numpy as np
import pprint


def load_dataset(filename="../../../dataset/clustering-examples/testSet.txt"):
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


def get_euclidean_distance(point1, point2):
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))


def create_rand_centroids(data, k):
    n = np.shape(data)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        min_value = np.min(data[:, j])
        cur_range = float(np.max(data[:, j]) - min_value)
        centroids[:, j] = min_value + cur_range * np.random.rand(k, 1)
    return centroids


def k_means(data, k, distance_measure=get_euclidean_distance, centroids_creation=create_rand_centroids):
    """
    Implementation of k-means algorithm.

    :param data:
    :param k:
    :param distance_measure:
    :param centroids_creation:
    :return:
    """
    m = np.shape(data)[0]
    assign = np.mat(np.zeros((m, 2)))
    centroids = centroids_creation(data, k)
    cluster_changed = True
    while cluster_changed:
        cluster_changed = False
        # assign i into the closest cluster
        for i in range(m):
            min_dist, min_idx = np.inf, -1
            for c in range(k):
                dist = distance_measure(centroids[c, :], data[i, :])
                if dist < min_dist:
                    min_dist, min_idx = dist, c
            if assign[i, 0] != min_idx:
                cluster_changed = True
            assign[i, :] = min_idx, min_dist**2
        print('Current centroids:')
        print(centroids)
        # update centroids with their mean values
        for c in range(k):
            p_in_c = data[np.nonzero(assign[:, 0].A == c)[0]]
            centroids[c, :] = np.mean(p_in_c, axis=0)
    return centroids, assign


def bisecting_cluster(data, k, distance_measure=get_euclidean_distance):
    """
    Implementation of bisecting k-means algorithm: choose the cluster with the largest SSE and split it and
    then repeat until we get to the user-defined number (k) of clusters.

    :param data:
    :param k:
    :param distance_measure:
    :return:
    """
    m = np.shape(data)[0]
    assign = np.mat(np.zeros((m, 2)))
    init_centroid = np.mean(data, axis=0).tolist()[0]
    centroids = [init_centroid]

    # go over all the points in the dataset and calculate the error between that point and the initial centroid
    for j in range(m):
        assign[j, 1] = distance_measure(np.mat(init_centroid), data[j, :]) ** 2
    # enter the while loop, which splits clusters until we have the desired number of clusters
    while len(centroids) < k:
        min_err = np.inf
        # iterate over all the clusters and find the best cluster to  do binary split
        for c in range(len(centroids)):
            p_in_c = data[np.nonzero(assign[:, 0].A == c)[0]]
            # binary split this c-th cluster and record the new centroids & assignments of 'points in c'
            split_centroids, split_assign = k_means(p_in_c, 2, distance_measure)
            split_err = np.sum(split_assign[:, 1])
            # the squared error of 'points not in c'
            not_split_err = np.sum(assign[np.nonzero(assign[:, 0].A != c)[0], 1])
            # iteratively update the best cluster to split
            if split_err + not_split_err < min_err:
                best_cluster_to_split = c
                best_new_centroids = split_centroids
                best_assign = split_assign.copy()
                min_err = split_err + not_split_err
        print('The best centroid to split is: ', best_cluster_to_split)

        # update the cluster number of the two new clusters as 'cluster_original' and 'len(centroids)'
        # (we add a new cluster, thus the number increases)
        best_assign[np.nonzero(best_assign[:, 0].A == 0)[0], 0] = best_cluster_to_split
        best_assign[np.nonzero(best_assign[:, 0].A == 1)[0], 0] = len(centroids)
        assign[np.nonzero(assign[:, 0].A == best_cluster_to_split)[0], :] = best_assign
        # print('The length of best assignments is: ', len(best_assign))

        # update the position of centroids for the chosen cluster and the new created cluster
        centroids[best_cluster_to_split] = best_new_centroids[0, :]
        centroids.append(best_new_centroids[1, :])

    return centroids, assign


if __name__ == '__main__':
    # 1: test k-means
    # data_mat = np.mat(load_dataset())
    # k_centroids, assignments = k_means(data_mat, 4)
    # print(assignments)

    # 2: test bisecting cluster
    data_mat2 = np.mat(load_dataset("../../../dataset/clustering-examples/testSet2.txt"))
    bisect_centroids, assignments = bisecting_cluster(data_mat2, 3)
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(bisect_centroids)
