"""
This python script describes that how to load MNIST dataset from ubyte file. Figures of samples in MNIST are provided.
Actually MNIST can be downloaded through tf.keras.datasets.mnist.load_data(), and the cached dataset is stored in
~/.keras/datasets/mnist.npz' for macOS and Linux.

Author:
    Hailiang Zhao
"""
import os
import struct
import numpy as np
import matplotlib.pyplot as plt
from src.utils.global_settings import remove_avx_warning

# remove the AVX warning
remove_avx_warning()


def load_mnist(path, kind='train'):
    """
    Load MNIST dataset.

    :param path: the path where the MNIST dataset is stored
    :param kind: train ('train') or test ('t10k'), 'train' in default
    :return: numpy array stored MNIST dataset
    """
    labels_path = os.path.join(path, '%s-labels-idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images-idx3-ubyte' % kind)
    with open(labels_path, 'rb') as label_file:
        _, _ = struct.unpack('>II', label_file.read(8))
        labels = np.fromfile(label_file, dtype=np.uint8)

    with open(images_path, 'rb') as image_file:
        _, _, rows, cols = struct.unpack('>IIII', image_file.read(16))
        images = np.fromfile(image_file, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


if __name__ == '__main__':
    # process of MNIST dataset
    train_images, train_labels = load_mnist('../../dataset/MNIST/unpacked/', kind='train')
    test_images, test_labels = load_mnist('../../dataset/MNIST/unpacked/', kind='t10k')

    # show MNIST example
    fig, ax = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = train_images[train_labels == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.savefig('mnist.eps')
    plt.show()
    plt.close()
