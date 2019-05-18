"""
Remove the AVX warning when the used TensorFlow is not locally-compiled on your specific computer.

Author:
    Hailiang Zhao
"""
import os


def remove_avx_warning():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
