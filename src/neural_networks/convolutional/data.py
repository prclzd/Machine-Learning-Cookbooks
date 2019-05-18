"""
This script defines the class Data, which is the basic object for training and testing. This class is general for
Image Classification where the depth of image has to be 1.

Author:
    Hailiang Zhao
"""
import numpy as np


class Data(object):
    def __init__(self, images, labels, num_classes=10, one_hot=False, need_normalize=True, reshape=False):
        """
        Initialization.

        :param images: the image samples (numpy array) with shape [num_samples, rows, columns, color_depth]
        :param labels: the label of samples (numpy array) with shape [num_samples]
        :param num_classes: the number of classes/labels
        :param one_hot: whether one hot coding is used
        :param need_normalize: whether normalization is needed ([0, 255] -> [0., 1.]
        :param reshape: whether reshape the features of image into one-dimension vector
        """
        assert num_classes != 0, 'Number of classes can not be zero!'
        assert images.shape[0] == labels.shape[0], 'Number of images is not the same with number of labels!'
        self.__num_samples = images.shape[0]

        if reshape:
            assert images.shape[3] == 1, 'Currently we do not handle images whose depth > 1!'
            images = images.reshape(images.shape[0], images.shape[1] * images.shape[2])

        if need_normalize:
            images = images / 255.0
        # deep copy is not needed
        self.__images = images

        if one_hot:
            indices = np.arange(self.__num_samples) * num_classes
            one_hot_labels = np.zeros([self.__num_samples, num_classes])
            one_hot_labels.flat[indices + labels.ravel()] = 1
            self.__labels = one_hot_labels
        else:
            self.__labels = labels

    @property
    def images(self):
        return self.__images

    @property
    def labels(self):
        return self.__labels

    @property
    def num_samples(self):
        return self.__num_samples
