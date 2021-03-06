import tensorflow as tf
from tensorflow import keras
from loguru import logger
from utils import check_gpu, get_metadata, train
from cgan_model import CondGeneratorModel, CondDiscriminatorModel

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
        tf.one_hot(train_labels, 10, dtype=tf.float32),
        tf.expand_dims(train_labels, 1),
    )).shuffle(buffer_size=buf_size).batch(batch_size=batch_size)
    return train_set


def visualize_cond_mnist_sample(samples, ax, i):
    img = tf.clip_by_value(samples[0][i, :, :, 0]
                           * mean_val + mean_val, 0, 255)
    ax.set_title('Sample for {0}'.format(i))
    ax.imshow(img.numpy().astype(int), cmap='gray')
    ax.set_axis_off()


class CondMnistInputGenerator():
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim

    def next(self, sample_size):
        feat = tf.random.normal([sample_size, self.feat_dim])
        labels = tf.convert_to_tensor(
            [i % 10 for i in range(sample_size)], dtype=tf.int32)
        labels_onehot = tf.one_hot(labels, 10, dtype=tf.float32)
        return [feat, labels_onehot, tf.expand_dims(labels, 1)]


def train_cond_mnist(project_id, epoch, train_per_epoch, interval):
    check_gpu(logger)
    train(
        dataset=load_cond_mnist_dataset(
            project_id=project_id,
            buf_size=60000,
            batch_size=256,
        ),
        gen=CondGeneratorModel(),
        dis=CondDiscriminatorModel(),
        gen_opt=keras.optimizers.Adam(1e-4),
        dis_opt=keras.optimizers.Adam(1e-4),
        logger=logger,
        epochs=epoch,
        start_epoch=0,
        interval=interval,
        train_per_epoch=train_per_epoch,
        sample_size=3,
        batch_size=32,
        visualize=visualize_cond_mnist_sample,
        project_metadata=get_metadata(project_id=project_id),
        gen_input_generator=CondMnistInputGenerator(feat_dim=100),
    )
