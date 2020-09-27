import os
import tensorflow as tf


def check_gpu(logger):
    gpus = tf.config.list_physical_devices('GPU')
    logger.info('Found Tensorflow with version ' + str(tf.__version__))
    logger.info('Found GPU: ' + str(len(gpus)))
    logger.info('Check GPU done.')


def create_directory_if_not_exist(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)
