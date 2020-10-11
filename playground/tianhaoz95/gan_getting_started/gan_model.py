from tensorflow import keras

def build_generator_model():
    model = keras.Sequential()
    # From a vector of 7*7*256 to (7,7,256)
    model.add(keras.layers.Dense(7*7*256, use_bias=False, input_shape=(100,)))
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