import tensorflow as tf
from tensorflow import keras


class GeneratorModel_v1(keras.Model):
    def __init__(self):
        super(GeneratorModel_v1, self).__init__()
        # Expand 7*7*128 features into a (7,7,128) tensor
        self.dense_1 = keras.layers.Dense(7*7*255)
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.LeakyReLU()
        self.reshape_1 = keras.layers.Reshape((7, 7, 255))
        # Expand (10,) to (7,7,1)
        self.dense_2 = keras.layers.Dense(7*7*1)
        self.reshape_2 = keras.layers.Reshape((7, 7, 1))
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
        # Expand features to image channels
        feat_x = self.dense_1(feat_x)
        feat_x = self.bn_1(feat_x)
        feat_x = self.relu_1(feat_x)
        feat_x = self.reshape_1(feat_x)
        # Expand label input to a image channel
        label_x = self.dense_2(label)
        label_x = self.reshape_2(label_x)
        # Concat label and feature
        x = tf.concat([feat_x, label_x], axis=3)
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


class DiscriminatorModel_v1(keras.Model):
    def __init__(self):
        super(DiscriminatorModel_v1, self).__init__()
        self.expand_layer = keras.layers.Dense(28*28*1)
        self.reshape_layer = keras.layers.Reshape((28, 28, 1))
        self.conv_1 = keras.layers.Conv2D(
            64, (5, 5), strides=2, padding='same', input_shape=(28, 28, 1))
        self.relu_1 = keras.layers.LeakyReLU()
        self.drop_1 = keras.layers.Dropout(0.3)
        self.conv_2 = keras.layers.Conv2D(
            128, (5, 5), strides=2, padding='same')
        self.relu_2 = keras.layers.LeakyReLU()
        self.drop_2 = keras.layers.Dropout(0.3)
        self.flatten = keras.layers.Flatten()
        self.out = keras.layers.Dense(1)

    def call(self, inputs):
        images_x = inputs[0]
        labels = inputs[1]
        labels_x = self.expand_layer(labels)
        labels_x = self.reshape_layer(labels_x)
        x = tf.concat([images_x, labels_x], axis=3)
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)
        x = self.flatten(x)
        x = self.out(x)
        return x
