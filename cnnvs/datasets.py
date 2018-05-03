import abc

import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage import binary_dilation, gaussian_filter
from skimage import io
from sklearn import preprocessing


#
class DatasetBuilder:
    """
        This class is responsible for extracting and processing all parts for a dataset.
    """

    __metaclass__ = abc.ABCMeta
    random = np.random.RandomState(012)
    sigma = np.random.RandomState(110)
    normal = np.random.RandomState(13412)

    # centres = np.random.RandomState(4)

    def __init__(self, patient, station, vol_size, strides, testing=False):
        self._x = None
        self._y = None
        self._gt = None
        self._output_shape = None
        self._testing = testing

        self._patient = patient
        self._station = station
        self._vol_size = vol_size
        self._vol_margin = ()
        for dim in vol_size:
            self._vol_margin += (dim // 2,)
        self._strides = strides

    def load_data(self):
        raw_path = '/home/jamesneill/dataset/raw_volumes_eq/eq_{}.tiff'
        gt_path = '/home/jamesneill/dataset/cleaned_gt/cleaned_gt_{}.tiff'

        # load the raw and gt tiffs
        x_path = raw_path.format(self.patient_id())
        x_tiff = DatasetBuilder.import_tiff(x_path)

        y_path = gt_path.format(self.patient_id())
        y_tiff = DatasetBuilder.import_tiff(y_path, binarise=True, scale=False)

        # ONLY FOR UNET
        strides = []
        for index, _ in enumerate(self._strides):
            if self._strides[index] is None:
                strides.append(x_tiff.shape[index])
            else:
                strides.append(self._strides[index])
        self._strides = tuple(strides)
        self._vol_size = tuple(strides)
        assert (x_tiff.shape == y_tiff.shape)

        self._x = x_tiff
        self._y = y_tiff

    def prepare_data(self):

        # REMOVE FOR UNET?
        # print(self._x.shape)
        # self._x = DatasetBuilder.pad(self._x, self._vol_size)
        # self._y = DatasetBuilder.pad(self._y, self._vol_size)
        # print(self._x.shape)
        self._output_shape = self._x.shape

    @abc.abstractmethod
    def extract_data(self):
        pass

    @abc.abstractmethod
    def process(self):
        pass

    def dataset(self):
        return (self._x, self._y)

    ############################

    def output_shape(self):
        return self._output_shape

    def patient_id(self):
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
        return station_data[self._patient][self._station]

    @staticmethod
    def import_tiff(data_loc, binarise=False, transpose=(1, 2, 0), scale=True):
        # type: (str, bool, tuple, bool) -> np.ndarray
        """
        Imports .tiff file into ndarray
        :param data_loc: Location of file
        :param binarise: Ensures labels are binary
        :param transpose: Shape to transpose output
        :param scale: Scales volume to zero mean/unit variance
        :return: Import .tiff file as ndarray
        """
        if binarise and scale:
            raise ValueError('Scaling takes precedence over binarising, please remove unwanted flag.')

        # Import tiff data
        tiff_data = io.imread(data_loc).transpose(transpose)
        tiff_data[tiff_data == -1] = 0
        
        # Scale input volume to zero mean, unit variance, if desired
        if scale:
            tiff_data = DatasetBuilder.volume_scaler(tiff_data.astype('float32') / np.max(tiff_data * 1.0))
            output = tiff_data
        # binarise (0,1)
        elif binarise:
            tiff_data_binary = np.ones(tiff_data.shape)
            tiff_data_binary[tiff_data == 0] = 0
            output = tiff_data_binary
        else:
            output = tiff_data
        return output

    @staticmethod
    def volume_scaler(vol):
        # type: (np.ndarray) -> np.ndarray
        """
        :param vol: Volume to be scaled
        :return: Re-scaled volume
        """
        return np.reshape(preprocessing.scale(np.reshape(vol, -1)), vol.shape)

    @staticmethod
    def pad(data, vol_size):
        # type: (np.ndarray, tuple) -> np.ndarray
        """

        :param data:
        :param vol_size:
        :return:
        """
        padding = ()
        for axis, _ in enumerate(data.shape):
            # x = vol_size[axis] - (data.shape[axis] % vol_size[axis])
            # x = (x/2) + 1
            padding += ((vol_size[axis], vol_size[axis]),)
        data = np.pad(data, padding, 'constant', constant_values=(0, 0))
        return data


class PixelwiseBuilder(DatasetBuilder):

    def __init__(self, patient, station, vol_size, strides, testing=False):
        for dim in vol_size:
            if not dim % 2 == 1:
                raise ValueError('Invalid volume size, no central pixel to classify')

        super(PixelwiseBuilder, self).__init__(patient, station, vol_size, strides, testing)
        self._y_dilated = None
        self._vessel_map = None
        self._near_vessel_map = None
        self._background_map = None

        self._station_vessels = (68000, 8000, 14000, 12000)
        self._max_samples = self._station_vessels[station - 1] * 2
        self._index = 0

        # TODO: Can't remember why i've specifically declared this
        self._temp_x = None
        self._temp_y = None

    def extract_data(self):
        # dilate the ground truth
        self.dilate_y()

        # vessel map needs to built after near vessel, _y is modified in the process
        self.build_background_map()
        self.build_near_vessel_map()
        self.build_vessel_map()

        vessel_samples = min(len(self._vessel_map[self._vessel_map == 1]), self._max_samples / 2)
        near_vessel_samples = 2 * vessel_samples / 3
        background_samples = vessel_samples - near_vessel_samples

        total_samples = vessel_samples + near_vessel_samples + background_samples

        self._temp_x = np.ndarray((total_samples, 1,) + self._vol_size, dtype=np.float32)
        # fill with an invalid label to ensure not over allocating arrays i.e. validation accuracy will be 0
        self._temp_y = np.full((total_samples, 1,), -2, dtype=np.float32)

        self.extract_samples(self._vessel_map, vessel_samples)
        self.extract_samples(self._near_vessel_map, near_vessel_samples)
        self.extract_samples(self._background_map, background_samples)

        self._x = self._temp_x
        self._y = self._temp_y

    def process(self, rotate=True, reflect=False, intensity=False, noise='gaussian'):
        # roll the axis back for tensorflow
        self._x = np.rollaxis(self._x, 1, 5)
        print('Rotation?', rotate)
        print('Reflection?', reflect)

        # if rotate is True:
        #     print('rotating all volumes 180..')
        #
        #     shape = (1,) + self._x.shape
        #     x = np.empty(shape)
        #     y = np.empty((1, shape[1], 1))
        #
        #     for rotation in range(2, 3):
        #         x[rotation - 2] = np.rot90(self._x, rotation, (1, 2))
        #         y[rotation - 2] = self._y
        #
        #     x = np.vstack(x)
        #     y = np.vstack(y)
        #
        #     self._x = np.concatenate((self._x, x))
        #     self._y = np.concatenate((self._y, y))

        if rotate is True:
            print('rotating all volumes..')

            shape = (3,) + self._x.shape
            x = np.empty(shape)
            y = np.empty((4, shape[1], 1))

            for rotation in range(1, 4):
                x[rotation - 1] = np.rot90(self._x, rotation, (1, 2))
                y[rotation - 1] = self._y

            x = np.vstack(x)
            y = np.vstack(y)

            self._x = np.concatenate((self._x, x))
            self._y = np.concatenate((self._y, y))

        # if reflect is True:
        #     print('reflecting all volumes..')
        #
        #     x = np.fliplr(self._x[:])
        #     y = self._y
        #
        #     self._x = np.concatenate((self._x, x))
        #     self._y = np.concatenate((self._y, y))

        # if noise is 'gaussian':
        #
        #     shape = (2,) + self._x.shape
        #     x = np.empty(shape)
        #     y = np.empty((2, shape[1], 1))
        #
        #     for iteration in range(0, 2):
        #         for index, voxel in enumerate(self._x):
        #             # GRADIENT + NOISE?
        #             # sigma = self.sigma.random_sample()
        #             # gradient = np.zeros(voxel.shape)
        #             # gradient[:, 13, :] = 1
        #             # gradient = gaussian_filter(gradient, 27)
        #             # normalised = PixelwiseBuilder.normalise(gradient)
        #             # scaled = PixelwiseBuilder.scale(normalised)
        #             # noise = 0.005 * self.normal.normal(0, sigma, voxel.shape)
        #             # scaled += noise
        #             #
        #             # x[index] = voxel * scaled
        #             # y[index] = self._y[index]
        #
        #             # JUST NOISE
        #             x[iteration, index] = voxel + np.random.normal(0, 0.01, voxel.shape)
        #             y[iteration, index] = self._y[index]
        #
        #     x = np.vstack(x)
        #     y = np.vstack(y)

            self._x = np.concatenate((self._x, x))
            self._y = np.concatenate((self._y, y))

    ##################

    def dilate_y(self):
        self._y_dilated = binary_dilation(self._y, iterations=3).astype(self._y.dtype)

    def build_vessel_map(self):
        self._vessel_map = self.subsample(self._y)

    def build_near_vessel_map(self):
        near_vessel_map = self._y_dilated - self._y
        self._near_vessel_map = self.subsample(near_vessel_map)

    def build_background_map(self):
        background_map = np.zeros(self._y.shape)
        background_map[self._y_dilated == 0] = 1
        background_map = self.subsample(background_map)

        v = (np.max(self._vol_size) / 2) + 1

        # select central region based on volume size then pad it back with zeros
        # ensures that voxels extracted from the background are always within bounds of the array
        background_map = background_map[v:-v, v:-v, v:-v]
        self._background_map = np.pad(background_map, ((v, v), (v, v), (v, v)), mode='constant')

    def extract_samples(self, extraction_map, samples):
        vol_size = self._vol_size
        vol_margin = self._vol_margin

        x, y, z = np.where(extraction_map == 1)
        indices = np.random.choice(len(x), samples, replace=False)

        for pos in indices:
            voxel = self._x[x[pos] - vol_margin[0]:x[pos] + vol_margin[0] + 1,
                    y[pos] - vol_margin[1]:y[pos] + vol_margin[1] + 1,
                    z[pos] - vol_margin[2]:z[pos] + vol_margin[2] + 1]

            # if voxel.shape != vol_size:
            #     continue

            self._temp_x[self._index] = voxel
            self._temp_y[self._index] = self._y[x[pos], y[pos], z[pos]]
            self._index += 1

    def subsample(self, volume):
        voxel_centres = np.zeros(volume.shape)
        strides = self._strides

        voxel_centres[0::strides[0], 0::strides[1], 0::strides[2]] = 1
        voxel_centres[
        int(strides[0] / 2)::strides[0],
        int(strides[1] / 2)::strides[1],
        int(strides[2] / 2)::strides[2]
        ] = 1

        volume[voxel_centres == 0] = 0
        return volume

    @staticmethod
    def gaussian_shift(input, amount=0.04, sigma=5.0):
        centre = PixelwiseBuilder.centres(input.shape, amount)
        centre = np.rot90(centre)
        gaussian = gaussian_filter(centre, sigma)
        scaled = PixelwiseBuilder.scale(gaussian)

        return input * scaled

    @staticmethod
    def normalise(a):
        b = a - np.min(a)
        c = b / np.max(b)
        return c

    @staticmethod
    def scale(a, min=0.2, max=1.0):
        weights = PixelwiseBuilder.normalise(a)
        weights *= (max - min)
        return max - weights

    @staticmethod
    def centres(shape, amount):
        pos = [np.random.randint(0, x) for x in shape]
        zeros = np.zeros(shape)
        zeros[pos] = 1
        return zeros
        # return np.random.choice([1.0, 0.0], size=shape, p=[amount, 1 - amount])


class FullyConvolutionalBuilder(DatasetBuilder):

    def __init__(self, patients, station, vol_size, strides=None, testing=False):
        self._testing = testing
        super(FullyConvolutionalBuilder, self).__init__(patients, station, vol_size, strides, testing)

    def extract_data(self):
        def slice_data(data):
            # type: (np.ndarray) -> np.ndarray
            """
            Slices data into voxels defined by vol_size & stride
            :param data: data to be sliced
            :param vol_size: size of the 3 dimensional volumes to extract
            :param strides: specify the stride length of slicing
            :return: sliced data in 5 dimensional ndarray
            """

            vol_size = self._vol_size
            strides = self._strides

            # get the expected output shape after slicing & initialise np array
            num_strides = FullyConvolutionalBuilder.count_strides(data.shape, vol_size, strides)
            # output_shape = (np.prod(num_strides), 1,) + vol_size
            output_shape = (np.prod(num_strides),) + vol_size
            sliced_data = np.ndarray(output_shape, dtype=np.float32)

            # for each dimension slice data from appropriate ranges
            #       start index:    n * stride
            #         end index:    size + (n * stride)
            index = 0
            for x in range(0, num_strides[0]):
                for y in range(0, num_strides[1]):
                    for z in range(0, num_strides[2]):
                        sliced_data[index] = data[strides[0] * x:vol_size[0] + (strides[0] * x),
                                             strides[1] * y:vol_size[1] + (strides[1] * y),
                                             strides[2] * z:vol_size[2] + (strides[2] * z)]
                        index += 1
            return sliced_data

        self._x = slice_data(self._x)
        self._y = slice_data(self._y)

    def process(self):
        empty_indexes = []

        # TODO: This may not be feasible with UNet

        if not self._testing:
            for index in range(0, self._y.shape[0]):
                # if not add the index to empty_indexes
                if not np.any(self._y[index]):
                    empty_indexes.append(index)

            # delete empty_indexes from both X_train and Y_train
            self._x = np.delete(self._x, empty_indexes, axis=0)
            self._y = np.delete(self._y, empty_indexes, axis=0)

        # VNet adding channel dimension
        # self._x = np.reshape(self._x, self._x.shape + (1,))

        # image distortion
        # if not self._testing:
        #     x = np.empty(self._x.shape)
        #     y = np.empty(self._y.shape)
        #
        #     for index, _ in enumerate(self._x):
        #         elastic_x, elastic_y = FullyConvolutionalBuilder.elastic_transform(self._x[index], self._y[index])
        #         x[index] = elastic_x.reshape(self._x[index].shape)
        #         y[index] = elastic_y.reshape(self._x[index].shape)
        #
        #     self._x = np.concatenate((self._x, x))
        #     self._y = np.concatenate((self._y, y))

        # roll the axis back for tensorflow
        self._x = np.rollaxis(self._x, 1, -1)
        self._y = np.rollaxis(self._y, 1, -1)

    @staticmethod
    def elastic_transform(X, Y, alpha=0.1, sigma=1, random_state=None):
        X = X.squeeze()
        Y = Y.squeeze()

        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = X.shape

        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
        indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
        X = map_coordinates(X, indices, order=1).reshape(shape)
        Y = map_coordinates(Y, indices, order=1).reshape(shape)
        return X, Y

    @staticmethod
    def count_strides(volume, subvolume, stride):
        # type: (tuple, tuple, tuple) -> tuple
        """
            Calculates the number of strides in each dimension for a volume after slicing, similar to convolution operation.
            :param volume: shape of data being sliced
            :param subvolume: shape of the 3 dimensional volumes being extracted
            :param stride: shape of the strides in each dimension
            :return: output shape
        """

        shape = ()
        for axis, _ in enumerate(volume):
            shape += ((((volume[axis] - subvolume[axis]) / stride[axis]) + 1),)

        return shape


class Dataset:
    def __init__(self, builder, patients, station, vol_size, strides, testing=False):
        self._x = []
        self._y = []
        self._builders = []
        self.output_shape = None

        if isinstance(patients, int):
            patients = (patients,)

        for patient in patients:
            self._builders.append(builder(patient, station, vol_size, strides, testing))

    def construct(self):
        for builder in self._builders:
            builder.load_data()
            builder.prepare_data()
            builder.extract_data()
            builder.process()
            x, y = builder.dataset()
            self.output_shape = x.shape

            self._x.append(x)
            self._y.append(y)
        self._x = np.vstack(self._x)
        self._y = np.vstack(self._y)
        return self.data()

    def x(self):
        return self._x

    def y(self):
        return self._y

    def data(self):
        return self.x(), self.y()
