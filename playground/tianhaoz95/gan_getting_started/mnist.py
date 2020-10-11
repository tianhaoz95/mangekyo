from gan_model import build_discriminator_model, build_generator_model
import tensorflow as tf
from loguru import logger
from tensorflow import keras
from utils import check_gpu, train, get_metadata

mean_val = 255.0 / 2.0


class MnistInputGenerator():
    def __init__(self, feat_dim):
        self.feat_dim = feat_dim

    def next(self, sample_size):
        feat = tf.random.normal([sample_size, self.feat_dim])
        return feat


def load_mnist_dataset(project_id, buffer_size, batch_size):
    train_imgs = None
    if project_id == "fashion_mnist":
        (train_imgs, _), (_, _) = keras.datasets.fashion_mnist.load_data()
    if project_id == "mnist":
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


def train_mnist(project_id, epoch):
    check_gpu(logger)
    project_metadata = get_metadata(project_id)
    train(
        dataset=load_mnist_dataset(
            project_id=project_id,
            buffer_size=60000,
            batch_size=256),
        gen=build_generator_model(),
        dis=build_discriminator_model(),
        gen_opt=keras.optimizers.Adam(1e-4),
        dis_opt=keras.optimizers.Adam(1e-4),
        logger=logger,
        epochs=epoch,
        start_epoch=0,
        interval=20,
        train_per_epoch=300,
        sample_size=4,
        batch_size=32,
        visualize=visualize_mnist_sample,
        project_metadata=project_metadata,
        gen_input_generator=MnistInputGenerator(feat_dim=100),
    )
