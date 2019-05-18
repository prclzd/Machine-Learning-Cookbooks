"""
This script gives a simple example on that how to train an Recurrent Neural Networks (RNN).

Author:
    Hailiang Zhao
"""
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import argparse
from src.utils.global_settings import remove_avx_warning


remove_avx_warning()
# create parser
parser = argparse.ArgumentParser(description='Creating classifier')

# [step 1: define necessary flags]
# [define flags on recording 'trained models' and 'event logs']
tf.flags.DEFINE_integer('log_print_freq', 10, 'The frequency of log-print over an epoch.')

# [define necessary learning flags (batch size, step numbers, learning rate and decay frequency, etc.)]
tf.flags.DEFINE_integer('batch_size', 128, 'Number of samples per batch.')
tf.flags.DEFINE_integer('num_epochs', 3, 'Number of epochs for training.')
tf.flags.DEFINE_float('initial_learning_rate', 0.001, 'The initial learning rate for optimization.')
tf.flags.DEFINE_integer('seed', 111, 'seed')
tf.flags.DEFINE_integer('num_hidden_neurons', 128, 'Number of neurons in hidden layers of RNN.')

# [define flags on sequential information for RNN]
tf.flags.DEFINE_integer('step_size', 28, 'The step size for RNN.')
tf.flags.DEFINE_integer('input_size', 28, 'The input size of RNN.')
tf.flags.DEFINE_integer('output_size', 10, 'The output size of RNN.')

FLAGS = tf.flags.FLAGS

# reset the graph
tf.reset_default_graph()
tf.set_random_seed(FLAGS.seed)
np.random.seed(FLAGS.seed)

# [step 2: prepare train, test and test data]
mnist = input_data.read_data_sets('../../../dataset/MNIST/', reshape=False, one_hot=False)
# batches of train data can be obtained by next_batch() directly
test_image = mnist.test.images.reshape([-1, FLAGS.step_size, FLAGS.input_size])
test_label = mnist.test.labels

# [step 3: define tensors (parameters, placeholders and calculation operation of model, loss and accuracy]
image_place = tf.placeholder(tf.float32, shape=([None, FLAGS.step_size, FLAGS.input_size]), name='image')
label_place = tf.placeholder(tf.int32, shape=([None]), name='label')

# the construction of model
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=FLAGS.num_hidden_neurons)
output, state = tf.nn.dynamic_rnn(cell, image_place, dtype=tf.float32)

# the calculation operation of loss
logits = tf.layers.dense(state, FLAGS.output_size)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_place, logits=logits))

# the calculation operation of accuracy
optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.initial_learning_rate).minimize(loss)
prediction = tf.nn.in_top_k(logits, label_place, 1)
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

# [step 4: run the session]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_batches = mnist.train.num_examples // FLAGS.batch_size
    for epoch_idx in range(FLAGS.num_epochs):
        for batch_idx in range(num_batches):
            train_data_batch, train_label_batch = mnist.train.next_batch(FLAGS.batch_size)
            train_data_batch = train_data_batch.reshape([-1, FLAGS.step_size, FLAGS.input_size])
            sess.run(optimizer, feed_dict={image_place: train_data_batch, label_place: train_label_batch})
        train_loss_epoch, train_accuracy_epoch = sess.run(
            [loss, accuracy],
            feed_dict={image_place: train_data_batch, label_place: train_label_batch})
        print('Epoch: {}, train loss: {:.3f}, train accuracy: {:.3f}'.
              format(epoch_idx + 1, train_loss_epoch, train_accuracy_epoch))
    test_loss, test_accuracy = sess.run(
        [loss, accuracy],
        feed_dict={image_place: test_image, label_place: test_label})
    print('Test loss: {:.3f}, test accuracy: {.3f}'.format(test_loss, test_accuracy))
