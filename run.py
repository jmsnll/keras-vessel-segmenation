import argparse
import os
import warnings

import numpy as np
import tensorflow as tf
from keras import backend as K

from cnnvs import utils
from cnnvs.networks import UNet

epochs = 1
batch_size = 32
# vol_size = 27, 27, 27
vol_size = 64, 64, 3
# strides = 3, 3, 3
strides = 16, 16, 1

tag = 'unet-test'

# TODO: randomstate = np.random.RandomState
np.random.seed(1)


def run(training_data, target_station):
    test_data = list({1, 2, 3} - set(training_data))[0]
    session_name = utils.name(tag, vol_size, strides, epochs, training_data, target_station)
    cnn = UNet(session_name, vol_size, strides, epochs, batch_size)
    # cnn = Pixelwise(session_name, strides=strides, epochs=epochs)
    cnn.train(training_data, target_station, verbose=1)
    cnn.test(test_data, target_station)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target-gpu', help='Target Tensorflow GPU', default='0')
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    warnings.simplefilter('ignore', UserWarning)

    with tf.device('/gpu:' + args.target_gpu):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        session = tf.Session(config=config)
        K.set_session(session)

        # run(training_data=(3, 1), target_station=1)
        # run(training_data=(3, 1), target_station=2)
        run(training_data=(3, 1), target_station=3)
        # run(training_data=(3, 1), target_station=4)

        # run(training_data=(1, 2), target_station=1)
        # run(training_data=(1, 2), target_station=2)
        # run(training_data=(1, 2), target_station=3)
        # run(training_data=(1, 2), target_station=4)
        #
        # run(training_data=(2, 3), target_station=1)
        # run(training_data=(2, 3), target_station=2)
        # run(training_data=(2, 3), target_station=3)
        # run(training_data=(2, 3), target_station=4)
