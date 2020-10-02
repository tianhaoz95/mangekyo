import tensorflow as tf
import os
import matplotlib.pyplot as plt
from utils import check_gpu, create_directory_if_not_exist, restore_previous_checkpoint
from loguru import logger
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

proj_id = 'mnist'
buffer_size = 60000
batch_size = 256
epochs = 3
noise_dim = 100
mean_val = 255.0 / 2.0
interval = 5
sample_size = 4
output_dir = os.path.join('.', 'output', proj_id)
img_output_dir = os.path.join(output_dir, 'images')
ckpt_output_dir = os.path.join(output_dir, 'checkpoints')


def prepare_dataset(dataset_id):
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


def discriminator_loss(real, fake):
    real_loss = tf.losses.BinaryCrossentropy(
        from_logits=True)(tf.ones_like(real), real)
    fake_loss = tf.losses.BinaryCrossentropy(
        from_logits=True)(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


def generator_loss(fake):
    return tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake), fake)


@tf.function
def train_step(imgs, gen, dis, gen_opt, dis_opt):
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        gen_imgs = gen(noise)
        real_out = dis(imgs)
        fake_out = dis(gen_imgs)
        gen_loss = generator_loss(fake_out)
        dis_loss = discriminator_loss(real_out, fake_out)
    gen_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
    dis_grads = dis_tape.gradient(dis_loss, dis.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))
    dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))


def generate_img(gen, rand_input, epoch):
    preds = gen(rand_input)
    fig = plt.figure(figsize=(sample_size, 1))
    fig, axs = plt.subplots(sample_size, sample_size)
    for i in range(sample_size):
        for j in range(sample_size):
            axs[i, j].imshow(preds[i, :, :, 0] * mean_val +
                             mean_val, cmap='gray')
            axs[i, j].set_axis_off()
    save_loc = os.path.join(img_output_dir, 'epoch_{0}'.format(epoch))
    create_directory_if_not_exist(save_loc)
    plt.margins(0, 0)
    plt.savefig(os.path.join(save_loc, 'sample.png'))


def main():
    train_set = prepare_dataset(proj_id)
    check_gpu(logger)
    gen = build_generator_model()
    dis = build_discriminator_model()
    gen_opt = keras.optimizers.Adam(1e-4)
    dis_opt = keras.optimizers.Adam(1e-4)
    create_directory_if_not_exist(ckpt_output_dir)
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=dis_opt,
                                     generator=gen,
                                     discriminator=dis)
    manager = tf.train.CheckpointManager(
        checkpoint, ckpt_output_dir, max_to_keep=3)
    restore_previous_checkpoint(checkpoint, manager, ckpt_output_dir, logger)
    for e in range(epochs):
        logger.info('Start epoch {0}'.format(e))
        for img_batch in train_set:
            train_step(img_batch, gen, dis, gen_opt, dis_opt)
        if (e + 1) % interval == 0:
            generate_img(gen, tf.random.normal([sample_size**2, noise_dim]), e)
            manager.save()
    logger.info('Done')


if __name__ == '__main__':
    main()
