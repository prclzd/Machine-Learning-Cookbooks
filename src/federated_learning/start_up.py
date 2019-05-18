"""
This script test whether tensorflow_federated can be run correctly or not.

Author:
    Hailiang Zhao
"""
from __future__ import absolute_import, division, print_function
from src.utils.global_settings import remove_avx_warning
import numpy as np
import tensorflow as tf
from tensorflow_federated import python as tff


remove_avx_warning()

nest = tf.contrib.framework.nest
np.random.seed(0)
tf.compat.v1.enable_v2_behavior()
tff.federated_computation(lambda: 'Hello, World!')()
