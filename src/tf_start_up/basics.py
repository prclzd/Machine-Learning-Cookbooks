"""
This is the basics of using TensorFlow, where we learn how to do basic math operations and initialize them
with TensorFlow initializer.

Author:
    Hailiang Zhao
"""
import os
import tensorflow as tf
from tensorflow.python.framework import ops
from src.utils.global_settings import remove_avx_warning

# remove the AVX warning
remove_avx_warning()

# [step 1: define necessary flags]
# [define flags on recording 'trained models' and 'event logs']
tf.flags.DEFINE_string('log_dir', os.path.dirname(os.path.abspath(__file__)) + '/logs',
                       'Directory where event logs are written to.')

# store all elements in FLAG structure
FLAGS = tf.flags.FLAGS
# check whether the log_dir and checkpoint_dir are absolute directory
assert os.path.isabs(FLAGS.log_dir), 'You must assign absolute path for --train_path'

# [step 2: prepare train, test and test data]
# [step 3: define tensors (variables, constants, placeholders and calculation operation of model, loss and accuracy]
# basic math operations
a = tf.constant(5.0, name='a')
b = tf.constant(10.0, name='b')
x = tf.math.add(a, b, name='add')
y = tf.math.divide(a, b, name='divide')

# run the session
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print('output: ', sess.run([a, b, x, y]))

writer.close()

# create variables (all of them must be initialized or restored from saved variables!)
weights = tf.Variable(tf.random_normal([2, 3], stddev=0.1), name='weights')
biases = tf.Variable(tf.zeros([3]), name='biases')
custom_var = tf.Variable(tf.zeros([3]), name='custom')

# get all the variables' tensors and store them in a list
all_variables_list = ops.get_collection(ops.GraphKeys.GLOBAL_VARIABLES)
# use global initializer to initialize all variables at the same time
# the following code is equal to 'init_all_op = tf.variables_initializer(var_list=all_variables_list)'
init_all_op = tf.global_variables_initializer()

# initialize the specific variables
custom_variables_list = [weights, custom_var]
init_custom_op = tf.variables_initializer(var_list=custom_variables_list)

# initialize variables using existing variables
new_weights = tf.Variable(weights.initialized_value(), name='new_weights')
init_new_weights_op = tf.variables_initializer(var_list=[new_weights])


if __name__ == '__main__':
    # run the session
    with tf.Session() as sess:
        sess.run(init_all_op)
        print('Initialize all variables successfully!')
        sess.run(init_custom_op)
        print('Initialize custom variables successfully!')
        sess.run(init_new_weights_op)
        print('Initialize variables from exist variables successfully!')
