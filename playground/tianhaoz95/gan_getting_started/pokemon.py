import os
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from utils import get_metadata, train, check_gpu

mean_val = 255.0 / 2.0


def load_pokemon_dataset():
    img_gen = keras.preprocessing.image.ImageDataGenerator(
        data_format='channels_last'
    )
    dataset = keras.preprocessing.image.DirectoryIterator(
        os.path.join('.', 'dataset'),
        img_gen,
        target_size=(128, 128),
        classes=['pokemon'],
        color_mode='rgb',
        batch_size=32,
    )
    return dataset

def visualize_pokemon_sample(samples, ax, i):
    img = tf.clip_by_value(samples[i, :, :, :] * mean_val + mean_val, 0, 255)
    ax.imshow(img.numpy().astype(int))
    ax.set_axis_off()


def build_generator_model():
    model = keras.Sequential()
    # From a vector of 32*32*64 to (32, 32, 64)
    model.add(keras.layers.Dense(16*16*64, use_bias=False, input_shape=(256,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((16, 16, 64)))
    # From (16, 16, 64) to (16, 16, 32)
    model.add(keras.layers.Conv2DTranspose(
        32, (5, 5), strides=1, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (16, 16, 32) to (32, 32, 16)
    model.add(keras.layers.Conv2DTranspose(
        16, (5, 5), strides=2, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (32, 32, 16) to (64, 64, 8)
    model.add(keras.layers.Conv2DTranspose(
        8, (5, 5), strides=2, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (64, 64, 8) to (128, 128, 3)
    model.add(keras.layers.Conv2DTranspose(
        3, (5, 5), strides=2, padding='same', use_bias=False))
    return model


def build_discriminator_model():
    model = keras.Sequential()
    # Outputs (128, 128, 64)
    model.add(keras.layers.Conv2D(64, (5, 5), strides=2,
                                  padding='same', input_shape=(128, 128, 3)))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))
    # Outputs (64, 64, 128)
    model.add(keras.layers.Conv2D(128, (5, 5), strides=2,
                                  padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))
    # Outputs (32, 32, 256)
    model.add(keras.layers.Conv2D(256, (5, 5), strides=2,
                                  padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dense(1))
    return model


def build_discriminator_model_with_resnet50():
    model = keras.Sequential()
    resnet50 = keras.applications.ResNet50(
        input_shape=(128, 128, 3), include_top=False)
    resnet50.trainable = False
    model.add(resnet50)
    model.add(keras.layers.Dense(128))
    model.add(keras.layers.Dense(64))
    model.add(keras.layers.Dense(1))
    return model


def train_pokemon():
    check_gpu(logger)
    project_id = 'pokemon'
    project_metadata = get_metadata(project_id)
    train(
        dataset=load_pokemon_dataset(),
        gen=build_generator_model(),
        dis=build_discriminator_model_with_resnet50(),
        gen_opt=keras.optimizers.Adam(1e-4),
        dis_opt=keras.optimizers.Adam(1e-4),
        logger=logger,
        epochs=2000,
        start_epoch=0,
        interval=10,
        train_per_epoch=64,
        sample_size=4,
        noise_dim=256,
        batch_size=32,
        mean_val=255.0/2.0,
        visualize=visualize_pokemon_sample,
        project_metadata=project_metadata,
    )
