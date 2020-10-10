import os
import semver
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_batch_ops import batch


def get_metadata(project_id):
    output_dir = os.path.join('.', 'output', project_id)
    image_mode = 'unknown'
    if project_id == 'mnist' or project_id == 'fashion_mnist':
        image_mode = 'greyscale'
    if project_id == 'pokemon':
        image_mode = 'rgb'
    return {
        'img_output_dir': os.path.join(output_dir, 'images'),
        'ckpt_output_dir': os.path.join(output_dir, 'checkpoints'),
        'image_mode': image_mode,
        'batch_size': 64
    }


def check_gpu(logger):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
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


def generate_img(gen, gen_input, epoch, sample_size, img_output_dir, visualize):
    preds = gen(gen_input)
    fig = plt.figure(figsize=(sample_size, 1))
    fig, axs = plt.subplots(sample_size, sample_size)
    for i in range(sample_size):
        for j in range(sample_size):
            visualize(preds, axs[i, j], i * sample_size + j)
    save_loc = os.path.join(img_output_dir, 'epoch_{0}'.format(epoch+1))
    create_directory_if_not_exist(save_loc)
    plt.margins(0, 0)
    plt.savefig(os.path.join(save_loc, 'sample.png'))
    plt.close('all')


def discriminator_loss(real, fake):
    real_loss = tf.losses.BinaryCrossentropy(
        from_logits=True)(tf.ones_like(real), real)
    fake_loss = tf.losses.BinaryCrossentropy(
        from_logits=True)(tf.zeros_like(fake), fake)
    return real_loss + fake_loss


def generator_loss(fake):
    return tf.losses.BinaryCrossentropy(from_logits=True)(tf.ones_like(fake), fake)


@tf.function
def train_step(imgs, gen, dis, gen_opt, dis_opt, batch_size, gen_input_generator):
    gen_input = gen_input_generator.next(batch_size)
    with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        gen_imgs = gen(gen_input)
        real_out = dis(imgs)
        fake_out = dis(gen_imgs)
        gen_loss = generator_loss(fake_out)
        dis_loss = discriminator_loss(real_out, fake_out)
    gen_grads = gen_tape.gradient(gen_loss, gen.trainable_variables)
    dis_grads = dis_tape.gradient(dis_loss, dis.trainable_variables)
    gen_opt.apply_gradients(zip(gen_grads, gen.trainable_variables))
    dis_opt.apply_gradients(zip(dis_grads, dis.trainable_variables))


def train(dataset, gen, dis, gen_opt, dis_opt, logger,
          epochs, start_epoch, interval, sample_size,
          batch_size, train_per_epoch, visualize,
          project_metadata, gen_input_generator):
    ckpt_output_dir = project_metadata['ckpt_output_dir']
    create_directory_if_not_exist(ckpt_output_dir)
    checkpoint = tf.train.Checkpoint(generator_optimizer=gen_opt,
                                     discriminator_optimizer=dis_opt,
                                     generator=gen,
                                     discriminator=dis)
    manager = tf.train.CheckpointManager(
        checkpoint, ckpt_output_dir, max_to_keep=3)
    restore_previous_checkpoint(checkpoint, manager, ckpt_output_dir, logger)
    max_epoch = start_epoch + epochs
    for e in range(start_epoch, max_epoch):
        step_cnt = 0
        for batch in dataset:
            if step_cnt % 25 == 0:
                logger.info(
                    'Epoch {0}/{1}: step {2}'.format(e, max_epoch, step_cnt))
            train_step(
                imgs=batch,
                gen=gen,
                dis=dis,
                gen_opt=gen_opt,
                dis_opt=dis_opt,
                batch_size=batch_size,
                gen_input_generator=gen_input_generator)
            step_cnt += 1
            if step_cnt > train_per_epoch:
                break
        if (e + 1) % interval == 0:
            generate_img(
                gen=gen,
                gen_input=gen_input_generator.next(sample_size**2),
                epoch=e,
                sample_size=sample_size,
                img_output_dir=project_metadata['img_output_dir'],
                visualize=visualize)
            manager.save()
    logger.info('Done')
