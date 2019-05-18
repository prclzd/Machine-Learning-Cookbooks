"""
This script defines two transactions: train and test.

Author:
    Hailiang Zhao
"""
import tensorflow as tf
import os
from src.utils.plot_progress_bar import plot_progress_bar


def train(**keywords):
    """
    The general train function with evaluation enabled.

    :param keywords:
        -> :param sess: current running TensorFlow session
        -> :param saver: current running TensorFlow saver
        -> :param tensors: a dict where used tensors saved, including:
            -> cost: the tensor (operation) who records the calculation of loss
            -> accuracy: the tensor (operation) who records the calculation of accuracy
            -> train_op: the tensor (operation) for train operation
            -> global_step: the tensor (operation) who stores the global step
            -> train_summary_op: the serialized [summary] protocol buffers of train operation (during each batch)
            -> train_summary_op_epoch: the serialized [summary] protocol buffers of train operation during each epoch
            -> validate_summary_op: the serialized [summary] protocol buffers of validate operation
            -> feature_place: the [placeholder] where images stored
            -> label_place: the [placeholder] where label stored
            -> dropout_param: the [placeholder] where dropout parameter stored
        -> :param data: structural dataset used in model training and testing
        -> :param log_dir: the log directory for summary of train and test
        -> :param fine_tuned: whether fine-tuned model is obtained or not
        -> :param num_epochs: number of epochs
        -> :param is_validating: whether validate during each training epoch is required or not
        -> :param checkpoint_dir: the checkpoint directory where fine-tuned model stored
        -> :param batch_size: the size (number of samples) of each batch
    :return: no return
    """
    # [step 5.2.1: define the summary writers for train and evaluate]
    train_summary_dir = os.path.join(keywords['log_dir'], 'summaries', 'train')
    train_summary_writer = tf.summary.FileWriter(train_summary_dir)
    train_summary_writer.add_graph(keywords['sess'].graph)

    validate_summary_dir = os.path.join(keywords['log_dir'], 'summaries', 'validate')
    validate_summary_writer = tf.summary.FileWriter(validate_summary_dir)
    validate_summary_writer.add_graph(keywords['sess'].graph)

    # [step 5.2.2: restore current model if it is fine-tuned]
    checkpoint_prefix = 'model'
    if keywords['fine_tuned']:
        keywords['saver'].restore(keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
        print('Fine-tuned model has been restored!')

    # [step 5.2.3: loop the training process over epochs]
    print('Training...')
    for epoch_idx in range(keywords['num_epochs']):
        train_num_batches = int(keywords['data'].train.images.shape[0] / keywords['batch_size'])
        # (1) go through each batch
        for batch_idx in range(train_num_batches):
            # (2) obtain the training data and label for current batch
            start_idx = batch_idx * keywords['batch_size']
            end_idx = (batch_idx + 1) * keywords['batch_size']
            train_feature_batch = keywords['data'].train.images[start_idx:end_idx]
            train_label_batch = keywords['data'].train.labels[start_idx:end_idx]

            # (3) run back-propagation optimization based on SGD algorithms
            train_loss_batch, _, train_summaries_batch, train_step = keywords['sess'].run(
                [keywords['tensors']['cost'],
                 keywords['tensors']['train_op'],
                 keywords['tensors']['train_summary_op'],
                 keywords['tensors']['global_step']],
                feed_dict={keywords['tensors']['feature_place']: train_feature_batch,
                           keywords['tensors']['label_place']: train_label_batch,
                           keywords['tensors']['dropout_param']: 0.5})

            # (4) write the optimization summary of this batch into train summary writers
            train_summary_writer.add_summary(train_summaries_batch, global_step=train_step)

            # (5) plot progress bar of (the rate of up-to-now trained batches to all batches)
            progress = float(batch_idx + 1) / train_num_batches
            plot_progress_bar(progress=progress, num_epochs=epoch_idx + 1, loss_batch=train_loss_batch)

        # (6) write the training summary of this epoch into writers
        train_summary_op_epoch = keywords['tensors']['train_summary_op_epoch']
        train_summaries_epoch = keywords['sess'].run(
            train_summary_op_epoch,
            feed_dict={keywords['tensors']['feature_place']: train_feature_batch,
                       keywords['tensors']['label_place']: train_label_batch,
                       keywords['tensors']['dropout_param']: 1.})
        train_summary_writer.add_summary(train_summaries_epoch, global_step=train_step)

        # (7) validate current model on the validation data instantly
        # (validate on a huge data directly may induce memory error, thus validation dataset usually is not large)
        if keywords['is_validating']:
            print('Validating...')
            validate_accuracy_epoch, validate_summaries_epoch = keywords['sess'].run(
                [keywords['tensors']['accuracy'], keywords['tensors']['validate_summary_op']],
                feed_dict={keywords['tensors']['feature_place']: keywords['data'].validation.images,
                           keywords['tensors']['label_place']: keywords['data'].validation.labels,
                           keywords['tensors']['dropout_param']: 1.})
            print('Epoch ' + str(epoch_idx + 1) + ', validate accuracy = ' + '{:.5f}'.format(validate_accuracy_epoch))

            # write the validating summary of this epoch into writers
            current_step = tf.train.global_step(keywords['sess'], keywords['tensors']['global_step'])
            validate_summary_writer.add_summary(validate_summaries_epoch, global_step=current_step)

    # [step 5.2.4: save current model into checkpoint directory
    model_saved_name = keywords['saver'].save(
        keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
    print('Model saved in: %s' % model_saved_name)


def test(**keywords):
    """
    The general test function.

        :param keywords:
        -> :param sess: current running TensorFlow session
        -> :param saver: current running TensorFlow saver
        -> :param tensors: a dict stored used tensors, including:
            -> accuracy: the tensor (operation) who records the calculation of accuracy
            -> feature_place: the placeholder where images stored
            -> label_place: the placeholder where label stored
            -> dropout_param: the placeholder where dropout parameter stored
        -> :param checkpoint_dir: the checkpoint directory where fine-tuned model stored
        -> :param batch_size: the size (number of samples) of each batch
    :return: no return
    """
    checkpoint_prefix = 'model'
    keywords['saver'].restore(keywords['sess'], os.path.join(keywords['checkpoint_dir'], checkpoint_prefix))
    print('Restore the fine-tuned model for test...')
    print('Testing...')
    test_num_batches = int(keywords['data'].test.images.shape[0] / keywords['batch_size'])
    test_accuracy = 0
    for batch_idx in range(test_num_batches):
        # obtain the testing data and label for current batch
        start_idx = batch_idx * keywords['batch_size']
        end_idx = (batch_idx + 1) * keywords['batch_size']
        test_feature_batch = keywords['data'].test.images[start_idx:end_idx]
        test_label_batch = keywords['data'].test.labels[start_idx:end_idx]
        test_accuracy_batch, test_loss_batch, test_summaries_batch, test_step = keywords['sess'].run(
            [keywords['tensors']['accuracy'],
             keywords['tensors']['cost'],
             keywords['tensors']['validate_summary_op'],
             keywords['tensors']['global_step']],
            feed_dict={keywords['tensors']['feature_place']: test_feature_batch,
                       keywords['tensors']['label_place']: test_label_batch})
        test_accuracy += test_accuracy_batch

        progress = float(batch_idx + 1) / test_num_batches
        # the test process just need one epoch
        plot_progress_bar(progress=progress, num_epochs=1, loss_batch=test_loss_batch)
    test_accuracy = test_accuracy / float(test_num_batches)
    print('Final test accuracy is %.2f%%' % test_accuracy)
