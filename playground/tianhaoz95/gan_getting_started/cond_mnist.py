import tensorflow as tf
from tensorflow import keras
from loguru import logger
from utils import check_gpu, get_metadata, train

mean_val = 255.0 / 2.0


def load_cond_mnist_dataset(project_id, buf_size, batch_size):
    train_imgs = None
    train_labels = None
    if 'cond_fashion_mnist' in project_id:
        (train_imgs, train_labels), (_, _) = keras.datasets.fashion_mnist.load_data()
    elif 'cond_mnist' in project_id:
        (train_imgs, train_labels), (_, _) = keras.datasets.mnist.load_data()
    else:
        raise NotImplementedError(
            'Project with ID {0} not implemented'.format(project_id))
    train_imgs = (train_imgs - mean_val) / mean_val
    train_set = tf.data.Dataset.from_tensor_slices((
        tf.expand_dims(train_imgs, 3),
        tf.one_hot(train_labels, 10, dtype=tf.float32)
    )).shuffle(buffer_size=buf_size).batch(batch_size=batch_size)
    return train_set


def visualize_cond_mnist_sample(samples, ax, i):
    img = tf.clip_by_value(samples[0][i, :, :, 0] * mean_val + mean_val, 0, 255)
    ax.imshow(img.numpy().astype(int), cmap='gray')
    ax.set_axis_off()

class CondMnistInputGenerator():
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim

    def next(self, sample_size):
        feat = tf.random.normal([sample_size, self.feat_dim])
        label = tf.one_hot([i % 10 for i in range(sample_size)], 10, dtype=tf.float32)
        return [feat, label]


class GeneratorModel_v2(keras.Model):
    def __init__(self):
        super(GeneratorModel_v1, self).__init__()
        # Expand 7*7*128 features into a (7,7,128) tensor
        self.dense_1 = keras.layers.Dense(7*7*255)
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
        x = tf.concat([feat_x, label])
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
        super(DiscriminatorModel_v1, self).__init__()
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
        x = tf.concat([x, labels])
        x = self.out(x)
        return x

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


def generator_factory(project_id):
    if 'v1' in project_id:
        return GeneratorModel_v1()
    elif 'v2' in project_id:
        return GeneratorModel_v2()
    else:
        raise NotImplemented('Project {0} not found.'.format(project_id))

def discriminator_factory(project_id):
    if 'v1' in project_id:
        return DiscriminatorModel_v1()
    elif 'v2' in project_id:
        return DiscriminatorModel_v2()
    else:
        raise NotImplemented('Project {0} not found.'.format(project_id))

def train_cond_mnist(project_id):
    check_gpu(logger)
    train(
        dataset=load_cond_mnist_dataset(
            project_id=project_id, 
            buf_size=60000, 
            batch_size=256,
        ),
        gen=generator_factory(project_id),
        dis=discriminator_factory(project_id),
        gen_opt=keras.optimizers.Adam(1e-4),
        dis_opt=keras.optimizers.Adam(1e-4),
        logger=logger,
        epochs=5000,
        start_epoch=0,
        interval=100,
        train_per_epoch=300,
        sample_size=4,
        batch_size=32,
        mean_val=mean_val,
        visualize=visualize_cond_mnist_sample,
        project_metadata=get_metadata(project_id=project_id),
        gen_input_generator=CondMnistInputGenerator(feat_dim=100),
    )
