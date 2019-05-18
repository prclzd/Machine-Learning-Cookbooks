"""
This script defines the function about that how to plot the progress bar for the training.

Author:
    Hailiang Zhao
"""
import sys


def plot_progress_bar(progress, num_epochs, loss_batch):
    """
    This function plot the progress bar of an epoch training.

    :param progress: a float value between [0, 1] indicates the progress
    :param num_epochs: number of training epochs
    :param loss_batch: the loss of a batch training
    :return: no return
    """
    assert 0 <= progress <= 1, 'Progress is unqualified!'
    bar_length = 30
    status = ''
    if progress >= 1:
        progress = 1
        status = '\r\n'
    pos = int(round(bar_length * progress))
    bar = [str(num_epochs), '#' * pos, '-' * (bar_length - pos), progress * 100, loss_batch, status]
    output = '\rEpoch {0[0]} {0[1]}{0[2]} {0[3]:.2f}% loss = {0[4]:.3f} {0[5]}'.format(bar)
    sys.stdout.write(output)
    sys.stdout.flush()
