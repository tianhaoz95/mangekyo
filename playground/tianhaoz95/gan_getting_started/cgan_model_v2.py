import tensorflow as tf
from tensorflow import keras


class GeneratorModel_v2(keras.Model):
    def __init__(self):
        super(GeneratorModel_v2, self).__init__()
        # Expand 7*7*128 features into a (7,7,128) tensor
        self.dense_1 = keras.layers.Dense(7*7*256)
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.LeakyReLU()
        self.reshape_1 = keras.layers.Reshape((7, 7, 256))
        # From (7,7,256) to (7,7,128)
        self.convt_1 = keras.layers.Conv2DTranspose(
            128, (5, 5), strides=1, padding='same', use_bias=False)
        self.convt_bn_1 = keras.layers.BatchNormalization()
        self.convt_relu_1 = keras.layers.LeakyReLU()
        # From (7,7,128) to (14,14,64)
        self.convt_2 = keras.layers.Conv2DTranspose(
            64, (5, 5), strides=2, padding='same', use_bias=False)
        self.convt_bn_2 = keras.layers.BatchNormalization()
        self.convt_relu_2 = keras.layers.LeakyReLU()
        # From (14,14,64) to (28,28,1)
        self.convt_out = keras.layers.Conv2DTranspose(
            1, (5, 5), strides=2, padding='same', use_bias=False)

    def call(self, inputs):
        feat_x = inputs[0]
        label = inputs[1]
        x = tf.concat([feat_x, label], axis=-1)
        # Expand features to image channels
        x = self.dense_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = self.reshape_1(x)
        # From (7,7,256) to (7,7,128)
        x = self.convt_1(x)
        x = self.convt_bn_1(x)
        x = self.convt_relu_1(x)
        # From (7,7,128) to (14,14,64)
        x = self.convt_2(x)
        x = self.convt_bn_2(x)
        x = self.convt_relu_2(x)
        # From (14,14,64) to (28,28,1)
        x = self.convt_out(x)
        return [x, label]


class DiscriminatorModel_v2(keras.Model):
    def __init__(self):
        super(DiscriminatorModel_v2, self).__init__()
        self.conv_1 = keras.layers.Conv2D(
            64, (5, 5), strides=2, padding='same', input_shape=(28, 28, 1))
        self.relu_1 = keras.layers.LeakyReLU()
        self.drop_1 = keras.layers.Dropout(0.3)
        self.conv_2 = keras.layers.Conv2D(
            128, (5, 5), strides=2, padding='same')
        self.relu_2 = keras.layers.LeakyReLU()
        self.drop_2 = keras.layers.Dropout(0.3)
        self.flatten = keras.layers.Flatten()
        self.dense_1 = keras.layers.Dense(128)
        self.dense_2 = keras.layers.Dense(64)
        self.out = keras.layers.Dense(1)

    def call(self, inputs):
        images_x = inputs[0]
        labels = inputs[1]
        x = self.conv_1(images_x)
        x = self.relu_1(x)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)
        x = self.flatten(x)
        x = tf.concat([x, labels], axis=-1)
        x = self.out(x)
        return x
