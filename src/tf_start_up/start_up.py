"""
This is the start-up of using TensorFlow.

Author:
    Hailiang Zhao
"""
import os
import tensorflow as tf
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
# [step 3: define tensors (variables, constants, placeholders and calculating operation of model, loss and accuracy]
welcome = tf.constant('Welcome to TensorFlow world!')


# run the session
with tf.Session() as sess:
    # write the summaries into logs
    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_dir), sess.graph)
    print('output: ', sess.run(welcome))
# close the writer
writer.close()
