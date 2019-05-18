"""
This python script describes that how to use TensorFlow to implement the Multi-Layer Perceptron (MLP) model for
Multi-class Classification.

Author:
    Hailiang Zhao
"""
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data


# ['define' flags for model records]
tf.flags.DEFINE_string('log_path', os.path.dirname(os.path.abspath(__file__)) + '/logs',
                       'Directory where event logs are written to.')
tf.flags.DEFINE_string('checkpoint_path', os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
                       'Directory where checkpoints are written to.')
tf.flags.DEFINE_integer('max_num_checkpoints', 10, 'Maximum number of checkpoints kept.')
tf.flags.DEFINE_integer('log_steps', 50, 'Number of steps per each display.')

# ['define' necessary learning flags (batch size, step numbers, learning rate and train or test)]
tf.flags.DEFINE_integer('batch_size', 128, 'Number of samples per batch.')
tf.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training.')
tf.flags.DEFINE_float('initial_learning_rate', 0.01, 'The initial learning rate for optimization.')
tf.flags.DEFINE_float('decay_factor', 0.95, 'The decay factor of learning rate.')
tf.flags.DEFINE_integer('num_classes', 10, 'The default number of classes of MNIST dataset.')
tf.flags.DEFINE_integer('num_epochs_per_decay', 1, 'The number of epochs passed to decay learning rate.')

# ['define' status flags for training]
tf.flags.DEFINE_boolean('is_evaluation', True, 'Whether or not the model should be evaluated.')
tf.flags.DEFINE_boolean('is_training', False, 'Training or testing.')
tf.flags.DEFINE_boolean('fine_tuning', False, 'Fine tuning is desired or not.')
tf.flags.DEFINE_boolean('online_test', True, 'Online test is used or not.')
tf.flags.DEFINE_boolean('allow_soft_placement', True,
                        'Automatically put the variables on CPU if there is no GPU support.')
tf.flags.DEFINE_boolean('log_device_placement', False,
                        'Demonstrate that which variables are on which devices.')

FLAGS = tf.flags.FLAGS

if not os.path.isabs(FLAGS.log_path):
    raise ValueError('You must assign absolute path for --train_path')
if not os.path.isabs(FLAGS.checkpoint_path):
    raise ValueError('You must assign absolute path for --checkpoint_path')


if __name__ == '__main__':
    # load MNIST dataset
    mnist = input_data.read_data_sets("../dataset/MNIST", reshape=True, one_hot=True)
    train_data = mnist.train.images
    train_label = mnist.train.labels
    test_data = mnist.test.images
    test_label = mnist.test.labels

    # get dataset info (to calculate decay steps)
    num_train_samples = train_data.shape[0]
    num_features = train_data.shape[1]

    # the relationship between graph and session?
    graph = tf.Graph()
    with graph.as_default():
        # ['define' placeholders and variables/constants]
        global_step = tf.Variable(0, name='global_step', trainable=False)
        decay_steps = int(num_train_samples / FLAGS.batch_size * FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(
            FLAGS.initial_learning_rate, global_step, decay_steps, FLAGS.decay_factor,
            staircase=True, name='exponential_decay_learning_rate')
        image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
        label_place = tf.placeholder(tf.float32, shape=([None, FLAGS.num_classes]), name='label')
        dropout_param = tf.placeholder(tf.float32)

        # ['define' multi-layer perceptron model: two hidden layer]
        # layer 1
        layer1 = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=250, scope='fc-1')
        # layer 2
        layer2 = tf.contrib.layers.fully_connected(inputs=layer1, num_outputs=250, scope='fc-2')
        # softmax layer
        logits_pre_softmax = tf.contrib.layers.fully_connected(inputs=layer2,
                                                               num_outputs=FLAGS.num_classes, scope='fc-3')
        # ['define' loss]
        softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_pre_softmax,
                                                                              labels=label_place))
        # ['define' accuracy]
        accuracy = tf.reduce_mean(tf.cast(
            tf.equal(tf.argmax(logits_pre_softmax, 1), tf.argmax(label_place, 1)), tf.float32))

        # ['define' optimizer]
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # gradient update
        with tf.name_scope('train_scope'):
            grads = optimizer.compute_gradients(softmax_loss)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)

        # ['define' summaries and merge them together]
        tf.summary.scalar("loss", softmax_loss, collections=['train', 'test'])
        tf.summary.scalar('accuracy', accuracy, collections=['train', 'test'])
        tf.summary.scalar('global_step', global_step, collections=['train'])
        tf.summary.scalar('learning_rate', learning_rate, collections=['train'])
        summary_train_op = tf.summary.merge_all('train')
        summary_test_op = tf.summary.merge_all('test')

        # config and start the session
        session_conf = tf.ConfigProto(allow_soft_placement=FLAGS.allow_soft_placement,
                                      log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(graph=graph, config=session_conf)

        # everything is defined, now we start training!
        with sess.as_default():
            saver = tf.train.Saver(max_to_keep=FLAGS.max_num_checkpoints)

            # ['run' initialization]
            sess.run(tf.global_variables_initializer())

            # ['define' summary writers for train/test]
            train_summary_dir = os.path.join(FLAGS.log_path, 'summaries', 'train')
            train_summary_writer = tf.summary.FileWriter(train_summary_dir)
            train_summary_writer.add_graph(sess.graph)

            test_summary_dir = os.path.join(FLAGS.checkpoint_path, 'summaries', 'test')
            test_summary_writer = tf.summary.FileWriter(test_summary_dir)
            test_summary_writer.add_graph(sess.graph)

            # restore current model if allowed
            checkpoint_prefix = 'model'
            if FLAGS.fine_tuning:
                saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
                print('Model restored!')

            # loop over the batches for running
            for epoch in range(FLAGS.num_epochs):
                total_batch_training = int(num_train_samples / FLAGS.batch_size)
                for batch_num in range(total_batch_training):
                    start_idx = batch_num * FLAGS.batch_size
                    end_idx = (batch_num + 1) * FLAGS.batch_size
                    train_data_batch, train_label_batch = train_data[start_idx:end_idx], train_label[start_idx:end_idx]

                    # ['run' optimization]
                    batch_loss, _, train_summaries, train_step = sess.run(
                        [softmax_loss, train_op, summary_train_op, global_step],
                        feed_dict={image_place: train_data_batch, label_place: train_label_batch, dropout_param: 0.5}
                    )

                    # write the summaries
                    train_summary_writer.add_summary(train_summaries, global_step=train_step)
                print('Epoch #' + str(epoch + 1) + ', train loss = ' + '{:.3f}'.format(batch_loss))

                # ['run' test]
                if FLAGS.online_test:
                    # refactor this part the code into batch mode to avoid memory error
                    # in case where test dataset is huge
                    test_accuracy_epoch, test_summaries = sess.run(
                        [accuracy, summary_test_op],
                        feed_dict={image_place: test_data, label_place: test_label, dropout_param: 1.0}
                    )
                    print('Epoch #' + str(epoch + 1) +
                          ', test accuracy = ' + '{:.2f} %'.format(100 * test_accuracy_epoch))

                    # get the value of current global_step
                    current_step = tf.train.global_step(sess, global_step)
                    # write the summaries
                    test_summary_writer.add_summary(test_summaries, global_step=current_step)

            # [save the model into checkpoints]
            save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print('Model saved in file: %s' % save_path)

            # ['run' test based on state-of-the-art model restored from checkpoint]
            saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print('Model restored...')
            # refactor this part the code into batch mode to avoid memory error
            # in case where test dataset is huge
            # total_test_accuracy = sess.run(accuracy, feed_dict={
            #     feature_place: test_data, label_place: test_label, dropout_param: 1.0
            # })
            # print('Final test accuracy is %.2f' % total_test_accuracy)
            total_batch_test = int(test_data.shape[0] / FLAGS.batch_size)
            test_accuracy = 0
            for batch_num in range(total_batch_test):
                start_idx = batch_num * FLAGS.batch_size
                end_idx = (batch_num + 1) * FLAGS.batch_size
                test_data_batch, test_label_batch = test_data[start_idx:end_idx], test_label[start_idx:end_idx]

                test_accuracy_batch, test_loss_batch, test_summaries_batch, test_step = sess.run(
                    [accuracy, softmax_loss, summary_test_op, global_step],
                    feed_dict={image_place: test_data_batch, label_place: test_label_batch}
                )
                test_accuracy += test_accuracy_batch

                test_summary_writer.add_summary(test_summaries_batch, global_step=test_step)
                print('Batch #' + str(batch_num + 1) +
                      ', test loss = ' + '{:.2f} %'.format(100 * test_accuracy_batch))

            # overall test
            test_accuracy_total = test_accuracy / float(total_batch_test)
            print('Total test accuracy = ' + '{:.2f} %'.format(100 * test_accuracy_total))
