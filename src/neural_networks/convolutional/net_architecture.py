"""
This script defines a function on constructing a Fully-convolutional Neural Network. Argument scopes ara enabled
for TensorBoard.

Author:
    Hailiang Zhao
"""
import tensorflow as tf


def create_net(images, num_classes=10, is_training=False, keep_prob=0.5, squeeze=True, scope='Net'):
    """
    Construct a Fully-connected Convolutional Neural Network.

    :param images: the input images
    :param num_classes: the number of classes
    :param is_training: indicate that training or not in dropout layer
    :param keep_prob: the keep probability of dropout layer
    :param squeeze: whether squeeze or not
    :param scope: the scope name for context manager of variables
    :return: the created network
    """
    with tf.variable_scope(scope, default_name='Net', values=[images, num_classes]) as sc:
        end_points_collection = sc.name + '_end_points'
        with tf.contrib.framework.arg_scope([tf.contrib.layers.conv2d, tf.contrib.layers.max_pool2d],
                                            outputs_collections=end_points_collection):
            # hidden layer 1
            net = tf.contrib.layers.conv2d(images, 32, [5, 5], scope='conv1')
            net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool1')

            # hidden layer 2
            net = tf.contrib.layers.conv2d(net, 64, [5, 5], scope='conv2')
            net = tf.contrib.layers.max_pool2d(net, [2, 2], 2, scope='pool2')

            # hidden layer 3
            net = tf.contrib.layers.conv2d(net, 1024, [7, 7], padding='VALID', scope='fc3')
            net = tf.contrib.layers.dropout(net, keep_prob, is_training=is_training, scope='dropout3')

            # output layer
            logits = tf.contrib.layers.conv2d(net, num_classes, [1, 1], activation_fn=None, scope='fc4')

            # return the collections as a dict
            end_points = tf.contrib.slim.utils.convert_collection_to_dict(end_points_collection)

            # squeeze the output layer logits spatially to eliminate extra dimensions
            if squeeze:
                logits = tf.squeeze(logits, [1, 2], name='fc4/squeezed')
                end_points[sc.name + '/fc4'] = logits
            return logits, end_points


def create_default_args(weight_decay=0.0005, is_training=False):
    """
    Define the default network arguments for the given set of list_ops.

    :param weight_decay: the decay rate of weight
    :param is_training: indicate that training or not in dropout layer
    :return: the scope of the net
    """
    if is_training:
        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d],
                padding='SAME',
                weights_regularizer=tf.contrib.slim.l2_regularizer(weight_decay),
                weights_initializer=tf.contrib.layers.variance_scaling_initializer(
                    factor=1.0,
                    mode='FAN_AVG',
                    uniform=False,
                    seed=None,
                    dtype=tf.float32),
                activation_fn=tf.nn.relu) as sc:
            return sc
    else:
        with tf.contrib.framework.arg_scope(
                [tf.contrib.layers.conv2d],
                padding='SAME',
                activation_fn=tf.nn.relu) as sc:
            return sc
