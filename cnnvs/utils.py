import numpy as np

# dictionary containing patient ids for each station
# data --> [patient] --> [station]
station_data = {
    1: {
        1: '63162_1',
        2: '63162_2',
        3: '63162_3',
        4: '63162_4'
    },
    2: {
        1: '70020_1',
        2: '70020_2',
        3: '70020_3',
        4: '70020_4'
    },
    3: {
        1: '69871_1',
        2: '69871_2',
        3: '56210_3',
        4: '56210_4'
    }
}


# TODO: Refactor this..
def name(prefix, vol_size, stride, epochs, training_data, target_station):
    """
    Generates names for based on todays date, used for when saving models.
    """
    # type: (str, int, int ,int, tuple, int) -> str

    return "{0}_{1[0]}x{1[1]}x{1[2]}_{2[0]}x{2[1]}x{2[2]}_{3}_{4[0]}{4[1]}_{5}".format(prefix, vol_size, stride,
                                                                                       epochs, training_data,
                                                                                       target_station)


def model_hcf(model_config):
    # type: (dict) -> int
    """
    Calculates the highest common factor a volume must divide into to be passed into the network.
    :param model_config: the config data of the model to be evaluated
    :return: highest common factor volume must divide into
    """

    # counts number of Conv3D layers
    conv_layers = list((x for x in model_config['layers'] if x['class_name'] == 'Conv3D'))
    # counts numer of layers with matching kernel sizes and strides
    contractions = list((x for x in conv_layers if x['config']['kernel_size'] == x['config']['strides']))

    # one is subtracted to account for the final softmax layer
    hcf = np.power(2, len(contractions) - 1)
    return hcf


def soft_segmentation_threshold(input_vol, thresh=20):
    """
    :param input_vol:
    :param thresh:
    :param plots:
    :return:
    """
    # Soft threshold of raw volume to reduce testing time
    soft_threshold = np.max(np.ndarray.flatten(input_vol)) / (thresh * 1.0)

    output_vol = np.zeros(input_vol.shape, dtype=bool)
    output_vol[input_vol >= soft_threshold] = True
    return output_vol


def keras_data_reformat(vol, output_shape=(68, 17, 17)):
    """ Reshape data array into shape expected by keras network.
    """

    # Reshape to format expected by keras
    reshaped_vol = vol.reshape(vol.shape[0], output_shape[0], output_shape[1], output_shape[2], 1).astype('float32')

    return reshaped_vol


def pad_data(data, subvolume):
    # type: (np.ndarray, tuple) -> np.ndarray
    """

    :param data:
    :param subvolume:
    :return:
    """
    padding = ()
    for axis, _ in enumerate(data.shape):
        padding += ((0, subvolume[axis] - (data.shape[axis] % subvolume[axis])),)
    return np.pad(data, padding, 'constant', constant_values=(0, 0))
