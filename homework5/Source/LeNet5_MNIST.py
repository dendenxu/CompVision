#! python
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl

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

# initailization for plotting and logging
# Setting up font for matplotlib
mpl.rc("font", family=["Josefin Sans", "Consolas", "Ubuntu", "Fira Code", "Inconsolata"], weight="medium", style="italic")
plt.style.use('dark_background')

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
            log.info(f"{len(gpus)} Physical GPUs, {len(logical_gpus)} Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            log.error(e)


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

    result_dir = "results"
    modelname = f"{result_dir}/models/{'dense' if use_dense else 'lenet5'}_{index}.pth"
    figname = f"{result_dir}/logs/{'dense' if use_dense else 'lenet5'}_{index}.svg"
    prediction_figname = f"prediction_{'dense' if use_dense else 'lenet5'}_{index}.svg"

    log.info("Loading MNIST dataset")
    ds_train, ds_test, ds_val, ds_info = load_mnist(batch_size)

    log.info("Constructing LeNet-5")

    construct_model = dense if use_dense else lenet5
    model = construct_model(input_shape, n_classes, lr)
    model.summary()

    log.info("Training...")
    history = model.fit(
        ds_train,
        epochs=n_epoch,
        validation_data=ds_val,
    )

    log.info("Plotting training history...")
    try:
        plot_training_history(history, figname)
    except Exception as e:
        log.error("Cannot plot the training history (are you on GUI?), however the figure is still saved to results/logs/{}_{}")

    log.info("Evaluating...")
    model.evaluate(ds_test)

    log.info("Saving model...")

    save_model(model, modelname)

    log.info("Predicting on random images...")
    predict_mnist(model, ds_test, prediction_figname)

    return model, history, ds_train, ds_test, ds_val, ds_info


def predict_mnist(model, ds_test, figname='prediction.svg'):
    try:
        ds_test = ds_test.unbatch()
    except Exception as e:
        log.error(e)
    plt.figure(figsize=(10, 10))
    plt.suptitle("Prediction & Ground Truth", fontweight="bold")
    for i, (img, label) in enumerate(ds_test.take(9)):
        plt.subplot(33*10 + i + 1)
        img = np.expand_dims(img, 0)
        result = model.predict(img)
        result = np.argmax(result)
        log.info(f"img shape: {img.shape}, label: {label}. predicted: {result}")
        plt.imshow(img.squeeze())
        plt.title(f"Prediction Index: {i}, Prediction: {result}, Truth: {label}")

    plt.tight_layout()
    plt.savefig(figname)
    plt.show()


def plot_training_history(res, figname, limit_acc_tick=False):
    def plot_key(key):
        length = len(res.history[key])
        inter = (interpolate.CubicSpline(np.linspace(0, length, length, endpoint=False), res.history[key]))(np.linspace(0, length, length*10, endpoint=False))
        plt.plot(inter, label=key, linewidth=2.5, alpha=0.7)

    plt.figure(figsize=(20, 10))
    plt.suptitle("Training History: loss & val_loss & acc & val_acc", fontweight="bold")
    plt.subplot(121)
    key = 'loss'
    plot_key(key)
    key = 'val_loss'
    plot_key(key)
    plt.legend(loc='right')

    plt.subplot(122)
    key = 'sparse_categorical_accuracy'
    plot_key(key)
    key = 'val_sparse_categorical_accuracy'
    plot_key(key)
    plt.legend(loc='right')
    if limit_acc_tick:
        plt.yticks(np.linspace(0, 1, 11, endpoint=True))

    plt.savefig(figname)
    plt.show()


if __name__ == "__main__":
    growth()
    main()
    main(True)
