import tensorflow as tf
from loguru import logger
from tensorflow import keras
from utils import check_gpu, train, get_metadata

mean_val = 255.0 / 2.0

def load_mnist_dataset(dataset_id, mean_val, buffer_size, batch_size):
    train_imgs = None
    if dataset_id == "fashion_mnist":
        (train_imgs, _), (_, _) = keras.datasets.fashion_mnist.load_data()
    if dataset_id == "mnist":
        (train_imgs, _), (_, _) = keras.datasets.mnist.load_data()
    train_imgs = train_imgs.reshape(
        train_imgs.shape[0], 28, 28, 1).astype('float32')
    train_imgs = (train_imgs - mean_val) / mean_val
    train_set = tf.data.Dataset.from_tensor_slices(
        train_imgs).shuffle(buffer_size).batch(batch_size)
    return train_set

def visualize_mnist_sample(samples, ax, i):
    img = tf.clip_by_value(samples[i, :, :, 0] * mean_val + mean_val, 0, 255)
    ax.imshow(img.numpy().astype(int), cmap='gray')
    ax.set_axis_off()

def build_generator_model():
    model = keras.Sequential()
    # From a vector of 7*7*256 to (7,7,256)
    model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(256,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((7, 7, 256)))
    # From (7, 7, 256) to (7, 7, 128)
    model.add(keras.layers.Conv2DTranspose(
        128, (5, 5), strides=1, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (7, 7, 128) to (14, 14, 64)
    model.add(keras.layers.Conv2DTranspose(
        64, (5, 5), strides=2, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (14, 14, 64) to (28, 28, 1)
    model.add(keras.layers.Conv2DTranspose(
        1, (5, 5), strides=2, padding='same', use_bias=False))
    return model

def build_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=2,
                            padding='same', input_shape=(28, 28, 1)))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    return model

def train_mnist(project_id):
    check_gpu(logger)
    buffer_size = 60000
    batch_size = 256
    mean_val = 255.0 / 2.0
    project_metadata = get_metadata(project_id)
    train(
        dataset=load_mnist_dataset(project_id, mean_val, buffer_size, batch_size),
        gen=build_generator_model(),
        dis=build_discriminator_model(),
        gen_opt=keras.optimizers.Adam(1e-4),
        dis_opt=keras.optimizers.Adam(1e-4),
        logger=logger,
        epochs=2000,
        start_epoch=0,
        interval=100,
        train_per_epoch=300,
        sample_size=4,
        noise_dim=256,
        batch_size=32,
        mean_val=mean_val,
        visualize=visualize_mnist_sample,
        project_metadata=project_metadata
    )
