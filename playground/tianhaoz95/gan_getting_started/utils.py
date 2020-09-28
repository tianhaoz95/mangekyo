import os
import semver
import tensorflow as tf


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
