import os
from loguru import logger
from tensorflow import keras
from utils import get_metadata, train, check_gpu

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

def build_generator_model():
    model = keras.Sequential()
    # From a vector of 32*32*64 to (32, 32, 64)
    model.add(keras.layers.Dense(32*32*64, use_bias=False, input_shape=(256,)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Reshape((32, 32, 64)))
    # From (32, 32, 64) to (32, 32, 32)
    model.add(keras.layers.Conv2DTranspose(
        32, (5, 5), strides=1, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (32, 32, 32) to (64, 64, 16)
    model.add(keras.layers.Conv2DTranspose(
        16, (5, 5), strides=2, padding='same', use_bias=False))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.LeakyReLU())
    # From (64, 64, 16) to (128, 128, 3)
    model.add(keras.layers.Conv2DTranspose(
        3, (5, 5), strides=2, padding='same', use_bias=False))
    return model

def build_discriminator_model():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(64, (5, 5), strides=2,
                            padding='same', input_shape=(128, 128, 3)))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, (5, 5), strides=2, padding='same'))
    model.add(keras.layers.LeakyReLU())
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(1))
    return model

def main():
    check_gpu(logger)
    project_id = 'pokemon'
    project_metadata = get_metadata(project_id)
    dataset = load_pokemon_dataset()
    train(
        dataset=dataset,
        gen=build_generator_model(),
        dis=build_discriminator_model(),
        gen_opt=keras.optimizers.Adam(1e-4),
        dis_opt=keras.optimizers.Adam(1e-4),
        logger=logger,
        ckpt_output_dir=project_metadata['ckpt_output_dir'],
        epochs=500,
        interval=5,
        sample_size=4,
        noise_dim=256,
        batch_size=32,
        mean_val=255.0/2.0,
        img_output_dir=project_metadata['img_output_dir']
    )

if __name__ == '__main__':
    main()