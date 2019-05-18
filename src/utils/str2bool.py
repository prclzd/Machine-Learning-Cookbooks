"""
This script defines a function on that how to convert a string from ('yes', 'true') to boolean variable 'True'.

Author:
    Hailiang Zhao
"""


def str2bool(judge):
    """
    Convert the input judgement (yes or no) into boolean variable (True or False).

    :param judge: the input judgement
    :return: True or False
    """
    return judge.lower() in ('yes', 'true')
