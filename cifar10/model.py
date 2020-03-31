import argparse
import sys

import numpy as np
import tensorflow as tf
from data import get_dataset, parse_dataset
from tensorflow.python.keras import Sequential
from tensorflow.python.keras import backend as kbe
from tensorflow.python.keras import callbacks
from tensorflow.python.keras import layers
from tensorflow.python.keras import losses
from tensorflow.python.keras import regularizers

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
kbe.set_session(session)


def create_model():
    model = Sequential(name='CIFAR-10')

    model.add(
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3),
                      padding='same', kernel_initializer='he_uniform',
                      kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform'))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_uniform',
                            kernel_regularizer=regularizers.l2()))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.25))
    model.add(layers.Dense(10))

    return model


def get_sgd_optimizer():
    return tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6,
                                   nesterov=True)


def get_available_gpus():
    from tensorflow.python.client import device_lib
    device_protos = device_lib.list_local_devices()
    return [i.name for i in device_protos if i.device_type == 'GPU']


def get_train_strategy(devices):
    try:
        strategy = tf.distribute.MirroredStrategy(devices=devices)
        return strategy
    except Exception as e:
        print(str(e))
        sys.exit(-1)


def get_callbacks():
    stop_on_nan_callback = callbacks.TerminateOnNaN()

    def decay(epoch):
        if epoch < 7:
            return 0.001
        else:
            return 0.001 * np.exp(0.1 * (7 - epoch))

    learning_rate_scheduler = callbacks.LearningRateScheduler(decay)

    return [stop_on_nan_callback, learning_rate_scheduler]


def main(arguments):
    print('===Start of program.===')

    tf.random.set_random_seed(12345)
    input_pattern = arguments.input_pattern
    output_name = arguments.output
    base_batch_size = 32
    available_gpus = get_available_gpus()
    total_batch_size = base_batch_size * len(available_gpus)
    dataset = parse_dataset(get_dataset(input_pattern),
                            batch_size=total_batch_size)
    strategy = get_train_strategy(available_gpus)

    callback = get_callbacks()
    with strategy.scope():
        optimizer = get_sgd_optimizer()
        model = create_model()
        cce_loss = losses.CategoricalCrossentropy(from_logits=True)
        print(model.summary())
        model.compile(loss=cce_loss, optimizer=optimizer)

        model.fit(dataset, epochs=20, callbacks=callback)
        if not output_name.endswith('.h5'):
            output_name += 'model.h5'
        model.save_weights(output_name, overwrite=True)
    print('===End of program.===')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input_pattern", type=str,
                        help="Input tfrecord data pattern", required=True)
    parser.add_argument("-o", "--output", type=str,
                        help="Output model file save name", required=True)
    args = parser.parse_args()

    main(args)
