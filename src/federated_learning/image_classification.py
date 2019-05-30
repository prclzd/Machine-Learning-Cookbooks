import collections
from six.moves import range
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.optimizer_v2 import gradient_descent
from tensorflow_federated import python as tff
from src.utils.global_settings import remove_avx_warning


remove_avx_warning()

nest = tf.contrib.framework.nest
np.random.seed(0)
tf.compat.v1.enable_v2_behavior()
tff.federated_computation(lambda: 'Hello, World!')()

# @test {"output": "ignore"}
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
len(emnist_train.client_ids)
# emnist_train.output_types, emnist_train.output_shapes


example_dataset = emnist_train.create_tf_dataset_for_client(emnist_train.client_ids[0])
example_element = iter(example_dataset).next()
example_element['label'].numpy()


NUM_EPOCHS = 10
BATCH_SIZE = 20
SHUFFLE_BUFFER = 500


def preprocess(dataset):
    def element_fn(element):
        return collections.OrderedDict([
            ('x', tf.reshape(element['pixels'], [-1])),
            ('y', tf.reshape(element['label'], [1])),
        ])

    return dataset.repeat(NUM_EPOCHS).map(element_fn).shuffle(
        SHUFFLE_BUFFER).batch(BATCH_SIZE)


# @test {"output": "ignore"}
preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = nest.map_structure(
    lambda x: x.numpy(), iter(preprocessed_example_dataset).next())

# sample_batch


def make_federated_data(client_data, client_ids):
    return [preprocess(client_data.create_tf_dataset_for_client(x)) for x in client_ids]


# @test {"output": "ignore"}
NUM_CLIENTS = 3
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]
federated_train_data = make_federated_data(emnist_train, sample_clients)


def create_compiled_keras_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(
            10, activation=tf.nn.softmax, kernel_initializer='zeros', input_shape=(784,))])

    def loss_fn(y_true, y_pred):
        return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred))

    model.compile(
        loss=loss_fn,
        optimizer=gradient_descent.SGD(learning_rate=0.02),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def model_fn():
    keras_model = create_compiled_keras_model()
    return tff.learning.from_compiled_keras_model(keras_model, sample_batch)


# @test {"output": "ignore"}
iterative_process = tff.learning.build_federated_averaging_process(model_fn)
state = iterative_process.initialize()
# @test {"timeout": 600, "output": "ignore"}
state, metrics = iterative_process.next(state, federated_train_data)
print('round  1, metrics={}'.format(metrics))
# @test {"skip": true}
for round_num in range(2, 101):
    state, metrics = iterative_process.next(state, federated_train_data)
    print('round {:2d}, metrics={}'.format(round_num, metrics))
