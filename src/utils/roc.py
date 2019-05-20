"""
This module provides the plot of Receiver Operating Characteristic (ROC) curve and the calculation of
Area Under the Curve (AUC).

Author:
    Hailiang Zhao
"""
import matplotlib.pyplot as plt
import numpy as np


def plot_roc(predict_strength, labels):
    """
    Plot ROC and return the calculated AUC.

    :param predict_strength: the strength of the classifier’s predictions
        Our classifier and our training functions generate this before they
        apply it to the np.sign() function.
    :param labels: the true label of each instance
    :return: the value of AUC
    """
    # at the beginning, every instance is classified into positive
    cursor = (1., 1.)
    y_sum = 0.
    pos_labels_num = sum(np.array(labels) == 1.)
    y_step = 1 / float(pos_labels_num)
    x_step = 1 / float(len(labels) - pos_labels_num)
    # sorted_strength is  from smallest (negative) to largest (positive), so we start at the
    # point (1.0,1.0) and draw to (0,0)
    sorted_strength = predict_strength.argsort()

    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for idx in sorted_strength.tolist()[0]:
        # the classified-to-be-positive instance is indeed positive, but we have to subtract it,
        # thus TP decreases, and the cursor takes a step down in the y direction
        if labels[idx] == 1.:
            delta_x, delta_y = 0, y_step
        else:
            delta_x, delta_y = x_step, 0
            y_sum += cursor[1]
        ax.plot([cursor[0], cursor[0]-delta_x], [cursor[1], cursor[1]-delta_y], c='b')
        cursor = (cursor[0]-delta_x, cursor[1]-delta_y)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False Positive Rate (FP)')
    plt.ylabel('True Positive Rate (TP)')
    plt.title('ROC curve')
    ax.axis([0, 1, 0, 1])
    plt.show()

    # the way we calculate auc is actually Calculus if x_step go to infinity small
    auc = y_sum * x_step
    print('The Area Under the Curve (AUC) is: ', auc)
    return auc
