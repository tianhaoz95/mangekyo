import os
import semver
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras


def check_gpu(logger):
    gpus = tf.config.list_physical_devices('GPU')
    logger.info('Found Tensorflow with version ' + str(tf.__version__))
    logger.info('Found GPU: ' + str(len(gpus)))
    logger.info('Check GPU done.')


def create_directory_if_not_exist(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def restore_previous_checkpoint(ckpt, ckpt_manager, ckpt_location, logger):
    if semver.compare(str(tf.__version__), "2.2.0") >= 0:
        ckpt_manager.restore_or_initialize()
    else:
        if os.path.exists(ckpt_location) and len(ckpt_manager.checkpoints) > 0:
            ckpt.restore(ckpt_manager.latest_checkpoint)
        else:
            logger.info('Checkpoint location not found. Skip.')
    logger.info('Restore checkpoint done.')


def load_pokemon_dataset():
    img_gen = keras.preprocessing.image.ImageDataGenerator(
        data_format='channels_last'
    )
    dataset = keras.preprocessing.image.DirectoryIterator(
        os.path.join('.', 'dataset'),
        img_gen,
        classes=['pokemon'],
        color_mode='grayscale'
    )
    return dataset


def generate_img(gen, rand_input, epoch, sample_size, mean_val, img_output_dir):
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


def discriminator_loss(real, fake):
    real_loss = tf.losses.BinaryCrossentropy(
        from_logits=True)(tf.ones_like(real), real)
    fake_loss = tf.losses.BinaryCrossentropy(
        from_logits=True)(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


def generator_loss(fake):
    return tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake), fake)


@tf.function
def train_step(imgs, gen, dis, gen_opt, dis_opt, batch_size, noise_dim):
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


def train(dataset, gen, dis, gen_opt, dis_opt, logger, ckpt_output_dir, epochs, interval, sample_size, noise_dim):
    check_gpu(logger)
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
        for img_batch in dataset:
            train_step(img_batch, gen, dis, gen_opt, dis_opt)
        if (e + 1) % interval == 0:
            generate_img(gen, tf.random.normal([sample_size**2, noise_dim]), e)
            manager.save()
    logger.info('Done')
