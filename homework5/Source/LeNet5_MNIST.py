#! python
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Activation, Conv2D, MaxPool2D, AveragePooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model, save_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
import tensorflow_datasets as tfds

import coloredlogs
import logging
coloredlogs.install("INFO")
log = logging.getLogger(__name__)


def growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def lenet5(input_shape, n_classes, lr):

    inputs = Input(shape=input_shape)
    model = Conv2D(6, kernel_size=(5, 5), strides=(1, 1), activation='tanh', input_shape=input_shape, padding="same")(inputs)
    model = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(model)
    model = Conv2D(16, kernel_size=(5, 5), strides=(1, 1), activation='tanh', padding='valid')(model)
    model = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(model)
    model = Flatten()(model)
    model = Dense(120, activation='tanh')(model)
    model = Dense(84, activation='tanh')(model)
    model = Dense(n_classes, activation='softmax')(model)
    model = Model(inputs, model)
    model.compile(
        optimizer=Adam(lr=lr),
        loss=SparseCategoricalCrossentropy(from_logits=True),
        metrics=[SparseCategoricalAccuracy()])
    return model


def dense(input_shape, n_classes, lr):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(n_classes)
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def load_mnist(batch_size=256):
    (ds_train, ds_test, ds_val), ds_info = tfds.load(
        'mnist',
        split=[
            tfds.Split.TRAIN,
            tfds.Split.TEST.subsplit(tfds.percent[:50]),
            tfds.Split.TEST.subsplit(tfds.percent[50:]),
        ],
        shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )

    def normalize_img(image, label):
        """Normalizes images: `uint8` -> `float32`."""
        return tf.cast(image, tf.float32) / 255., label

    # Applying normalization before `ds.cache()` to re-use it.
    # Note: Random transformations (e.g. images augmentations) should be applied
    # after both `ds.cache()` (to avoid caching randomness) and `ds.batch()` (for
    # vectorization [1]).
    ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_train = ds_train.cache()
    # For true randomness, we set the shuffle buffer to the full dataset size.
    ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
    # Batch after shuffling to get unique batches at each epoch.
    ds_train = ds_train.batch(batch_size)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_test = ds_test.batch(batch_size)
    ds_test = ds_test.cache()
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    ds_val = ds_val.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    ds_val = ds_val.batch(batch_size)
    ds_val = ds_val.cache()
    ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)

    return ds_train, ds_test, ds_val, ds_info


def main(use_dense=False, index=0):
    input_shape = (28, 28, 1)
    n_classes = 10
    batch_size = 256
    lr = 1e-3
    n_epoch = 20
    log.info("Loading MNIST dataset")
    ds_train, ds_test, ds_val, ds_info = load_mnist(batch_size)

    log.info("Constructing LeNet-5")
    if use_dense:
        model = dense(input_shape, n_classes, lr)
    else:
        model = lenet5(input_shape, n_classes, lr)
    model.summary()

    log.info("Training...")
    history = model.fit(
        ds_train,
        epochs=n_epoch,
        validation_data=ds_val,
    )

    plot_training_history(history, use_dense, index)

    log.info("Evaluating...")
    model.evaluate(ds_test)

    log.info("Saving model...")

    if use_dense:
        save_model(model, f"results/models/dense_{index}.pth")
    else:
        save_model(model, f"results/models/lenet5_{index}.pth")

    predict_mnist(ds_test)

    return model, history


def predict_mnist(ds_test):
    pass


def plot_training_history(res, use_dense, index):
    for key in res.history.keys():
        plt.plot(res.history[key], label=key)
    plt.legend(loc='upper right')
    plt.savefig(f"results/logs/{'dense' if use_dense else 'lenet5'}{index}.eps")
    plt.show()


if __name__ == "__main__":
    growth()
    main()
