"""
This python script describes that how to use TensorFlow to implement the Logistic Regression model.

Author:
    Hailiang Zhao
"""
import tensorflow as tf
import os
from src.utils.load_mnist import load_mnist

# define flags for storing logs
tf.flags.DEFINE_string(
    'log_path', os.path.dirname(os.path.abspath(__file__)) + '/logs',
    'Directory where event logs are written to.')
# define flags for storing checkpoints (fine-tuned models)
tf.flags.DEFINE_string(
    'checkpoint_path',
    os.path.dirname(os.path.abspath(__file__)) + '/checkpoints',
    'Directory where checkpoints are written to.')

tf.flags.DEFINE_integer('max_num_checkpoint', 10, 'Maximum number of checkpoints that TensorFlow will keep.')
tf.flags.DEFINE_integer('num_classes', 2, 'Number of model clones to deploy.')
tf.flags.DEFINE_integer('batch_size', 512, 'Number of model clones to deploy.')
tf.flags.DEFINE_integer('num_epochs', 10, 'Number of epochs for training.')

# define learning rate
tf.flags.DEFINE_float('initial_learning_rate', 0.001, 'Initial learning rate.')
tf.flags.DEFINE_float(
    'learning_rate_decay_factor', 0.95, 'Learning rate decay factor.')
tf.flags.DEFINE_float(
    'num_epochs_per_decay', 1, 'Number of epoch pass to decay learning rate.')

# define train parameters
tf.flags.DEFINE_boolean('is_training', False, 'Training/Testing.')
tf.flags.DEFINE_boolean('fine_tuning', False, 'Fine tuning is desired or not?.')
tf.flags.DEFINE_boolean('online_test', True, 'Fine tuning is desired or not?.')
tf.flags.DEFINE_boolean('allow_soft_placement', True,
                        'Automatically put the variables on CPU if there is no GPU support.')
tf.flags.DEFINE_boolean('log_device_placement', False, 'Demonstrate which variables are on what device.')

# store all elements in FLAG structure
FLAGS = tf.flags.FLAGS

if not os.path.isabs(FLAGS.log_path):
    raise ValueError('You must assign absolute path for --train_path')
if not os.path.isabs(FLAGS.checkpoint_path):
    raise ValueError('You must assign absolute path for --checkpoint_path')

# dataset pre-processing
# organize the dataset and feed it to associated dictionaries
data = dict()

train_images, train_labels = load_mnist('/Users/hliangzhao/Documents/Research/Dataset/MNIST', kind='train')
test_images, test_labels = load_mnist('/Users/hliangzhao/Documents/Research/Dataset/MNIST', kind='t10k')

constructed_train_idx = []
for i in range(len(train_images)):
    if train_labels[i] == 1 or train_labels[i] == 0:
        constructed_train_idx.append(i)
data['train/image'] = train_images[constructed_train_idx]
data['train/y'] = train_labels[constructed_train_idx]

constructed_test_idx = []
for i in range(len(test_images)):
    if test_labels[i] == 1 or test_labels[i] == 0:
        constructed_test_idx.append(i)
data['test/image'] = test_images[constructed_test_idx]
data['test/y'] = test_labels[constructed_test_idx]

# dimensions
dimensionality_train = data['train/image'].shape
num_train_samples = dimensionality_train[0]
num_features = dimensionality_train[1]

if __name__ == '__main__':
    # define graph
    graph = tf.Graph()
    with graph.as_default():
        # global step
        global_step = tf.Variable(0, name="global_step", trainable=False)

        # learning rate policy
        decay_steps = int(num_train_samples / FLAGS.batch_size *
                          FLAGS.num_epochs_per_decay)
        learning_rate = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps,
                                                   FLAGS.learning_rate_decay_factor, staircase=True,
                                                   name='exponential_decay_learning_rate')

        # place holders
        image_place = tf.placeholder(tf.float32, shape=([None, num_features]), name='image')
        label_place = tf.placeholder(tf.int32, shape=([None, ]), name='gt')
        label_one_hot = tf.one_hot(label_place, depth=FLAGS.num_classes, axis=-1)
        dropout_param = tf.placeholder(tf.float32)

        # model: A simple fully connected with two class and a softmax is equivalent to Logistic Regression.
        logits = tf.contrib.layers.fully_connected(inputs=image_place, num_outputs=FLAGS.num_classes, scope='fc')

        # loss
        with tf.name_scope('loss'):
            loss_tensor = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=label_one_hot))

        # accuracy
        prediction_correct = tf.equal(tf.argmax(logits, 1), tf.argmax(label_one_hot, 1))
        accuracy = tf.reduce_mean(tf.cast(prediction_correct, tf.float32))

        # define optimizer by its default values
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # 'train_op' is a operation that is run for gradient update on parameters.
        # Each execution of 'train_op' is a training step.
        # By passing 'global_step' to the optimizer, each time that the 'train_op' is run,
        # TensorFlow update the 'global_step' and increment it by one

        # gradient update
        with tf.name_scope('train_op'):
            gradients_and_variables = optimizer.compute_gradients(loss_tensor)
            train_op = optimizer.apply_gradients(gradients_and_variables, global_step=global_step)

        # run the session
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(graph=graph, config=session_conf)

        with sess.as_default():
            # The saver op.
            saver = tf.train.Saver()
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            # The prefix for checkpoint files
            checkpoint_prefix = 'model'
            # If fine-tuning flag in 'True' the model will be restored.
            if FLAGS.fine_tuning:
                saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
                print("Model restored for fine-tuning...")

            # go through the epochs
            test_accuracy = 0
            for epoch in range(FLAGS.num_epochs):
                total_batch_training = int(num_train_samples / FLAGS.batch_size)
                # go through the batches
                for batch_num in range(total_batch_training):
                    # get the training batches
                    start_idx = batch_num * FLAGS.batch_size
                    end_idx = (batch_num + 1) * FLAGS.batch_size
                    # Fit training using batch dataset
                    train_batch_data = data['train/image'][start_idx:end_idx]
                    train_batch_label = data['train/y'][start_idx:end_idx]

                    # Run optimization op (backprop) and Calculate batch loss and accuracy
                    # When the tensor tensors['global_step'] is evaluated, it will be incremented by one.
                    batch_loss, _, training_step = sess.run([loss_tensor, train_op, global_step],
                                                            feed_dict={image_place: train_batch_data,
                                                                       label_place: train_batch_label,
                                                                       dropout_param: 0.5})

                    # write the summaries into logs
                    writer = tf.summary.FileWriter(os.path.expanduser(FLAGS.log_path), sess.graph)

                print("Epoch " + str(epoch + 1) + ", Training Loss= " + "{:.5f}".format(batch_loss))

            # save the model checkpoints
            # create the path for saving the checkpoints
            if not os.path.exists(FLAGS.checkpoint_path):
                os.makedirs(FLAGS.checkpoint_path)
            # save the model
            save_path = saver.save(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print("Model saved in file: %s" % save_path)

            # test on test dataset
            # The prefix for checkpoint files
            checkpoint_prefix = 'model'
            # Restoring the saved weights.
            saver.restore(sess, os.path.join(FLAGS.checkpoint_path, checkpoint_prefix))
            print("Model restored...")

            # Evaluation of the model
            test_accuracy = 100 * sess.run(accuracy, feed_dict={
                image_place: data['test/image'],
                label_place: data['test/y'],
                dropout_param: 1.})

            print("Final Test Accuracy is %.2f %%" % test_accuracy)
