import tensorflow as tf
from tensorflow import keras
from loguru import logger
from utils import check_gpu

mean_val = 255.0 / 2.0


def load_cond_mnist_dataset(project_id, buf_size, batch_size):
    train_imgs = None
    train_labels = None
    if project_id == 'cond_fashion_mnist':
        (train_imgs, train_labels), (_, _) = keras.datasets.fashion_mnist.load_data()
    elif project_id == 'cond_mnist':
        (train_imgs, train_labels), (_, _) = keras.datasets.mnist.load_data()
    else:
        raise NotImplementedError(
            'Project with ID {0} not implemented'.format(project_id))
    train_imgs = (train_imgs - mean_val) / mean_val
    train_set = tf.data.Dataset.from_tensor_slices({
        'image': train_imgs,
        'label': tf.one_hot(train_labels, 10)
    }).shuffle(buffer_size=buf_size).batch(batch_size=batch_size)
    return train_set


class GeneratorModel_v1(keras.Model):
    def __init__(self):
        # Expand 7*7*128 features into a (7,7,128) tensor
        self.dense_1 = keras.layers.Dense(7*7*255)
        self.bn_1 = keras.layers.BatchNormalization()
        self.relu_1 = keras.layers.LeakyReLU()
        self.reshape = keras.layers.Reshape((7,7, 255))
        # Expand (10,) to (7,7,1)
        self.dense_2 = keras.layers.Dense(7*7*1)
        # From (7,7,256) to (7,7,128)
        self.convt_1 = keras.layers.Conv2DTranspose(128, (5, 5), strides=1, padding='same', use_bias=False)
        # From (7,7,128) to (14,14,64)


    def run(self, inputs):
        feat_x = inputs['feat']
        label_x = inputs['label']


class DiscriminatorModel_v1(keras.Model):
    def __init__(self):
        self.expand_layer = keras.layers.Dense(28*28*1)
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

    def run(self, inputs):
        images_x = inputs['image']
        labels = inputs['label']
        labels_x = self.expand_layer(labels)
        x = tf.stack([images_x, labels_x])
        x = self.conv_1(x)
        x = self.relu_1(x)
        x = self.drop_1(x)
        x = self.conv_2(x)
        x = self.relu_2(x)
        x = self.drop_2(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


def train_cond_mnist(project_id):
    check_gpu(logger)
    buf_size = 60000
    batch_size = 256
    train_set = load_cond_mnist_dataset(project_id, buf_size, batch_size)
