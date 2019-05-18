"""
This script runs the training and testing process based on defined functions and classes.

Author:
    Hailiang Zhao
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import collections
from src.neural_networks.convolutional import net_architecture
from src.neural_networks.convolutional.data import Data
from src.neural_networks.convolutional import transaction
from src.utils.global_settings import remove_avx_warning


remove_avx_warning()

# [step 1: define necessary flags]
# [(1) define flags on recording 'trained models' and 'event logs']
tf.flags.DEFINE_string('log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
                       'Directory where event logs are written to.')
tf.flags.DEFINE_string('checkpoint_dir', os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
                       'Directory where checkpoints are written to.')
tf.flags.DEFINE_integer('max_num_checkpoints', 10, 'Maximum number of checkpoints kept.')

# [(2) define necessary learning flags (batch size, step numbers, learning rate and decay frequency, etc.)]
tf.flags.DEFINE_integer('batch_size', 512, 'Number of samples per batch.')
tf.flags.DEFINE_integer('num_epochs', 2, 'Number of epochs for training.')
tf.flags.DEFINE_float('initial_learning_rate', 0.001, 'The initial learning rate for optimization.')
tf.flags.DEFINE_float('decay_factor', 0.95, 'The decay factor of learning rate.')
tf.flags.DEFINE_integer('num_classes', 10, 'The default number of classes of MNIST dataset.')
tf.flags.DEFINE_integer('num_epochs_per_decay', 1, 'The number of epochs passed to decay learning rate.')

# [(3) define status flags]
tf.flags.DEFINE_boolean('is_evaluation', True, 'Whether or not the model should be evaluated.')
tf.flags.DEFINE_boolean('is_training', False, 'Training or testing.')
tf.flags.DEFINE_boolean('is_validating', True, 'Online test (validate) is allowed or not.')
tf.flags.DEFINE_boolean('fine_tuned', False, 'Fine tuning is desired or not.')
tf.flags.DEFINE_boolean('allow_soft_placement', True,
                        'Automatically put the variables on CPU if there is no GPU support.')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Demonstrate that which variables are on which devices.')

FLAGS = tf.flags.FLAGS

# check whether the log_dir and checkpoint_dir are absolute directory
assert os.path.isabs(FLAGS.log_dir), 'You must assign absolute path for --train_path'
assert os.path.isabs(FLAGS.checkpoint_dir), 'You must assign absolute path for --checkpoint_path'

# [step 2: prepare train, test and test data]
mnist = input_data.read_data_sets('../../../dataset/MNIST/compressed', reshape=False, one_hot=False)
train = Data(mnist.train.images, mnist.train.labels, num_classes=10, one_hot=True, need_normalize=True, reshape=False)
validation = Data(mnist.validation.images, mnist.validation.labels,
                  num_classes=10, one_hot=True, need_normalize=True, reshape=False)
test = Data(mnist.test.images, mnist.test.labels,
            num_classes=10, one_hot=True, need_normalize=True, reshape=False)
data_collections = collections.namedtuple('data_collections', ['train', 'validation', 'test'])
data = data_collections(train=train, validation=validation, test=test)

# read basic information on each dimension of train data
train_info = data.train.images.shape
train_num_samples = train_info[0]
height = train_info[1]
width = train_info[2]
depth = train_info[3]

# [step 3: define operations and tensors, and store them in a data flow graph]
graph = tf.Graph()
with graph.as_default():
    # [step 3.1: define tensors, including variables, constants, placeholders]
    # (1) variables
    global_step = tf.Variable(initial_value=0, name='global_step', trainable=False)
    # the calculation of the exponentially-decayed learning rate
    learning_rate = tf.train.exponential_decay(
        FLAGS.initial_learning_rate,
        global_step=global_step,
        decay_steps=int(train_num_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay),
        decay_rate=FLAGS.decay_factor,
        staircase=True,
        name='exponential_decay_learning_rate')
    # (2) placeholders
    feature_place = tf.placeholder(tf.float32, shape=([None, height, width, depth]), name='feature')
    label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='label')
    dropout_param = tf.placeholder(tf.float32)
    # [step 3.2: define calculating operations on logits (model outputs), loss and accuracy
    # (1) by constructing network to obtain logits
    arg_scope = net_architecture.create_default_args(weight_decay=0.0005, is_training=FLAGS.is_training)
    with tf.contrib.framework.arg_scope(arg_scope):
        logits, end_points = net_architecture.create_net(
            feature_place,
            num_classes=FLAGS.num_classes,
            is_training=FLAGS.is_training,
            keep_prob=dropout_param)
    # (2) the calculating operation of loss
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_place))
    # (3) the calculating operation of accuracy
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(logits, 1),
            tf.argmax(label_place, 1)
        ), tf.float32))
    # (4) the train operation with a optimizer, which is an operation applying specific gradient methods
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    with tf.name_scope('train'):
        grads_and_vars = optimizer.compute_gradients(loss)
        train_op = optimizer.apply_gradients(grads_and_vars=grads_and_vars, global_step=global_step)

    # [step 4: define summaries and merge them]
    # (1) summary protocol for images: add three randomly chosen images from training set into summaries
    chosen_indices = np.random.randint(data.train.images.shape[0], size=(3,))
    tf.summary.image('images', data.train.images[chosen_indices], max_outputs=3, collections=['train_epoch'])
    # (2) summary protocol for histograms: add histograms of end points into summaries
    for idx in end_points:
        end_point = end_points[idx]
        tf.summary.scalar('sparsity/' + idx, tf.math.zero_fraction(end_point), collections=['train', 'validate'])
        tf.summary.histogram('activations/' + idx, end_point, collections=['train_epoch'])
    # (3) summary protocol for scalars: add loss and accuracy into summaries
    tf.summary.scalar("loss", loss, collections=['train', 'validate'])
    tf.summary.scalar("accuracy", accuracy, collections=['train', 'validate'])
    tf.summary.scalar("global_step", global_step, collections=['train'])
    tf.summary.scalar("learning_rate", learning_rate, collections=['train'])
    # merge summaries
    train_summary_op = tf.summary.merge_all('train')
    validate_summary_op = tf.summary.merge_all('validate')
    train_summary_op_epoch = tf.summary.merge_all('train_epoch')

    # collect tensors into dict
    tensor_keys = ['cost', 'accuracy', 'train_op', 'global_step',                        # operations
                   'feature_place', 'label_place', 'dropout_param',                      # placeholders
                   'train_summary_op', 'validate_summary_op', 'train_summary_op_epoch']  # serialized summary buffers
    tensor_values = [loss, accuracy, train_op, global_step,
                     feature_place, label_place, dropout_param,
                     train_summary_op, validate_summary_op, train_summary_op_epoch]
    tensors = dict(zip(tensor_keys, tensor_values))

    # [step 5: run the session]
    config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(graph=graph, config=config)
    with sess.as_default():
        saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)
        # [step 5.1: initialization]
        sess.run(tf.global_variables_initializer())
        # [step 5.2: train (train + evaluation)]
        transaction.train(sess=sess, saver=saver, tensors=tensors, data=data, log_dir=FLAGS.log_dir,
                          fine_tuned=FLAGS.fine_tuned, is_validating=FLAGS.is_validating, num_epochs=FLAGS.num_epochs,
                          checkpoint_dir=FLAGS.checkpoint_dir, batch_size=FLAGS.batch_size)
        # [step 5.3: test (final test)]
        transaction.test(sess=sess, saver=saver, tensors=tensors, data=data,
                         checkpoint_dir=FLAGS.checkpoint_dir, batch_size=FLAGS.batch_size)
