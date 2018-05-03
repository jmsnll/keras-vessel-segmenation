import abc
import cPickle as pickle
import os
import time

import keras
import numpy as np
from keras import Sequential, Input, Model
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.layers import Conv3D, MaxPooling3D, Flatten, Dense, Dropout, activations, Conv2D, Conv2DTranspose, \
    concatenate, Add, Concatenate, Conv3DTranspose
from keras.models import load_model
from keras.regularizers import l2
from tifffile import imsave

from cnnvs import losses, activations, metrics, utils
from datasets import Dataset, DatasetBuilder, PixelwiseBuilder, FullyConvolutionalBuilder


class Network:
    _builder_class = None

    __metaclass__ = abc.ABCMeta

    def __init__(self, session_name, vol_size, strides, epochs, batch_size):
        """
        Initialises the Network
        :param session_name: Name of the session
        :param vol_size: Size of the volume
        :param strides: Strides for extraction
        :param epochs: Number of epochs for training & testing
        :param batch_size: Batch size for training & testing
        """
        # type: (str, Union[int, tuple], Union[int, tuple], int, int) -> object
        self.session_name = session_name
        self.vol_size = vol_size
        self.strides = strides
        self.epochs = epochs
        self.batch_size = batch_size

        architecture = self.__class__.__name__

        if not os.path.isdir('/home/jamesneill/vessel_seg/session/{}/'.format(architecture)):
            print('Creating directories..')
            os.makedirs('/home/jamesneill/vessel_seg/session/{}/models/'.format(architecture))
            os.makedirs('/home/jamesneill/vessel_seg/session/{}/results/'.format(architecture))
            os.makedirs('/home/jamesneill/vessel_seg/session/{}/logs/'.format(architecture))

    def train(self, training_data, target_station, verbose=1):
        print(self.session_name)
        # print(training_data, target_station, self.vol_size, self.strides)

        dataset = Dataset(self._builder_class, training_data, target_station, self.vol_size, self.strides)
        (X_train, Y_train) = dataset.construct()

        model = self.model(summary=False)
        model.fit(X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, callbacks=self.callbacks(),
                  validation_split=0.05, verbose=verbose, shuffle=True)

    @abc.abstractmethod
    def model(self, summary=False):
        pass

    @abc.abstractmethod
    def test(self, test_data, target_station):
        pass

    # =====

    def callbacks(self):
        checkpoint_dir = '/home/jamesneill/vessel_seg/session/{}/models/{}.h5'.format(self.__class__.__name__,
                                                                                      self.session_name)
        checkpoint = ModelCheckpoint(checkpoint_dir, save_best_only=True)

        tensorboard_dir = '/home/jamesneill/vessel_seg/session/{}/logs/{}'.format(self.__class__.__name__,
                                                                                  self.session_name)
        tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0, write_graph=True,
                                  batch_size=self.batch_size)

        return [checkpoint, tensorboard]


class Pixelwise(Network):
    _builder_class = PixelwiseBuilder

    def __init__(self, session_name, vol_size=(27, 27, 27), strides=(1, 1, 1), epochs=15, batch_size=32):
        super(Pixelwise, self).__init__(session_name, vol_size, strides, epochs, batch_size)

    def model(self, l2_w=0.01, summary=True):
        # create sequential model
        model = Sequential()

        model.add(Conv3D(32, (3, 3, 3), activation='relu', padding='valid', input_shape=(self.vol_size + (1,)),
                         kernel_initializer='he_normal', kernel_regularizer=l2(l2_w)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(64, (3, 3, 3), activation='relu', padding='valid', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_w)))
        model.add(MaxPooling3D(pool_size=(2, 2, 2)))

        model.add(Conv3D(128, (3, 3, 3), activation='relu', padding='valid', kernel_initializer='he_normal',
                         kernel_regularizer=l2(l2_w)))

        # Fully connected layer: 1 output for classification
        model.add(Flatten())
        model.add(Dense(16, activation='sigmoid', kernel_initializer='he_normal'))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid', kernel_initializer='he_normal'))

        # Compile model with appropriate loss and optimizer functions
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        if summary:
            print(model.summary())

        return model

    def test(self, test_data, target_station):

        vol_size = 27
        vol_margin = vol_size / 2
        predict_block_size = 40000

        data_path = '/home/jamesneill/dataset/raw_volumes_eq/eq_{}.tiff'.format(
            utils.station_data[test_data][target_station])
        model_path = '/home/jamesneill/vessel_seg/session/{}/models/{}.h5'.format(self.__class__.__name__,
                                                                                  self.session_name)
        save_path = '/home/jamesneill/vessel_seg/session/{}/results/{}.pkl'.format(self.__class__.__name__,
                                                                                   self.session_name)

        print('Testing: {}'.format(model_path))

        model = load_model(model_path)

        raw_vol_test = DatasetBuilder.import_tiff(data_path)
        raw_vol_test_thresh = utils.soft_segmentation_threshold(raw_vol_test)
        raw_vol_crop_test_pad = np.pad(raw_vol_test, vol_margin, 'constant')
        sparse_gt_test_pad = np.pad(raw_vol_test_thresh, vol_margin, 'constant')

        step_counter = 0
        predict_subvol_list = []
        predict_test = []
        batch_counter = 0

        total_test_points = len(np.where(np.ndarray.flatten(raw_vol_test_thresh) == 1)[0])

        total_batches = np.ceil(total_test_points / (predict_block_size * 1.0))

        for x in range(vol_margin, raw_vol_crop_test_pad.shape[0] - vol_margin):
            for y in range(vol_margin, raw_vol_crop_test_pad.shape[1] - vol_margin):
                for z in range(vol_margin, raw_vol_crop_test_pad.shape[2] - vol_margin):
                    if sparse_gt_test_pad[x, y, z] == 1:
                        predict_subvol_list.append(raw_vol_crop_test_pad[x - vol_margin:x + vol_margin + 1,
                                                   y - vol_margin:y + vol_margin + 1,
                                                   z - vol_margin:z + vol_margin + 1])
                        step_counter += 1

                        if step_counter == predict_block_size:
                            X_test = utils.keras_data_reformat(np.asarray(predict_subvol_list),
                                                               (vol_size, vol_size, vol_size))
                            predict_test.append(model.predict(X_test, batch_size=self.batch_size))
                            predict_subvol_list = []
                            step_counter = 0

                            batch_counter += 1

                            print("Completed batch %d of %d" % (batch_counter, total_batches))

        if step_counter > 0:
            X_test = utils.keras_data_reformat(np.asarray(predict_subvol_list), (vol_size, vol_size, vol_size))
            predict_test.append(model.predict(X_test, batch_size=self.batch_size))

        _test = np.vstack(predict_test)

        _test_reconstructed = np.zeros(np.ndarray.flatten(raw_vol_test).shape)
        _test_reconstructed[np.where(np.ndarray.flatten(raw_vol_test_thresh) == 1)[0]] = np.squeeze(_test)
        _test_reshape = np.reshape(_test_reconstructed, raw_vol_test.shape)

        # Binarise results for final classification
        classification_thresh = 0.5
        _test_reshape[_test_reshape < classification_thresh] = 0
        _test_reshape[_test_reshape >= classification_thresh] = 1

        print('Saving output..')
        pickle.dump(_test_reshape, open(save_path, 'wb'))
        imsave('{}.tiff'.format(save_path), _test_reshape.astype(np.int16))
        print('Done!')


class FullyConvolutionalNetwork(Network):
    _builder_class = FullyConvolutionalBuilder

    def __init__(self, session_name, vol_size, strides, epochs, batch_size):
        super(FullyConvolutionalNetwork, self).__init__(session_name, vol_size, strides, epochs, batch_size)

    def model(self, summary=False):
        pass

    def test(self, test_data, target_station):
        """
        :param target_station:
        :param test_data:
        """

        model_path = '/home/jamesneill/vessel_seg/session/{}/models/{}.h5'.format(self.__class__.__name__,
                                                                                  self.session_name)

        save_path = '/home/jamesneill/vessel_seg/session/{}/results/{}.pkl'.format(self.__class__.__name__,
                                                                                   self.session_name)

        custom_objects = {
            'dsc': metrics.dsc,
            'iou': metrics.iou,
            'dsc_loss': losses.dsc,
            'prelu': activations.prelu
        }

        print('Testing: {}'.format(model_path))

        # Load model with custom objects suchas dice loss, mIoU etc.
        model = load_model(model_path, custom_objects=custom_objects)

        self.strides = (None, None, 1)
        dataset = Dataset(self._builder_class, test_data, target_station, self.vol_size, self.strides, testing=True)
        (X_test, _) = dataset.construct()
        output_shape = X_test.shape

        padding = []
        padding.append((0, 0))
        for axis in X_test.shape[1:-1]:
            padding.append((0, 16 - (axis % 16)))
        padding.append((0, 0))
        padding = tuple(padding)
        print(padding)

        X_test = np.pad(X_test, padding, 'constant', constant_values=(0, 0))

        rebuilt = model.predict(X_test)
        rebuilt = rebuilt[0:output_shape[0], 0:output_shape[1], 0:output_shape[2]]
        rebuilt = rebuilt[0:output_shape[0], 0:output_shape[1], 0:output_shape[2]]
        rebuilt = rebuilt[0:output_shape[0], 0:output_shape[1], 0:output_shape[2]]
        rebuilt = np.swapaxes(rebuilt, 2, 0)
        print('rebuilt.shape', rebuilt.shape)
        # index = 0
        # for x in range(0, (output_shape[0] / strides[0]) / 2):
        #     for y in range(0, (output_shape[1] / strides[1]) / 2):
        #         for z in range(0, (output_shape[2] / strides[2]) / 2):
        #             rebuilt[index] = predict_test[index]
        #
        #             # rebuilt[strides[0] * x:vol_size[0] + (strides[0] * x),
        #             # strides[1] * y:vol_size[1] + (strides[1] * y),
        #             # strides[2] * z:vol_size[2] + (strides[2] * z)] = predict_test[index]
        #
        #
        #             # strides[2] * z:vol_size[2] + (strides[2] * z)] += predict_test[index]
        #             index += 1

        # rebuilt -= np.min(rebuilt)
        # rebuilt /= np.max(rebuilt)
        # rebuilt -= np.mean(rebuilt) - 0.5

        # print('np.max(rebuilt)', np.max(rebuilt))
        # rebuilt /= np.max(rebuilt)

        classification_thresh = 0.5
        rebuilt[rebuilt < classification_thresh] = 0
        rebuilt[rebuilt >= classification_thresh] = 1

        print('Saving output..')
        pickle.dump(rebuilt, open(save_path, 'wb'))
        imsave('{}.tiff'.format(save_path), rebuilt.astype(np.int16))
        print('Done!')

        # print('predict_test.shape:', predict_test.shape)
        # predict_test = np.reshape(predict_test, builder._output_shape)
        # print('predict_test.shape:', predict_test.shape)
        # pickle.dump(predict_test, open(save_path, 'wb'))


class UNet(FullyConvolutionalNetwork):

    def __init__(self, session_name, vol_size=(64, 64, 1), strides=(32, 32, 1), epochs=10, batch_size=32):
        super(UNet, self).__init__(session_name, vol_size, strides, epochs, batch_size)

    def model(self, summary=False):
        """
        Builds a U-Net fully convolutional model
        :param summary: Show the model summary
        :param concat_axis: Axis for concatination layers
        :return: Compiled Keras model
        """
        inputs = Input(shape=(None, None, 1))
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
        pool1 = Conv2D(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv1)

        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
        conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
        pool2 = Conv2D(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv2)

        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
        conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
        pool3 = Conv2D(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv3)

        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
        conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
        pool4 = Conv2D(512, (2, 2), strides=(2, 2), activation='relu', padding='same')(conv4)

        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
        conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

        up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
        conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

        up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
        conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

        up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
        conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

        up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
        conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

        conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

        model = Model(inputs=[inputs], outputs=[conv10])
        model.compile(optimizer='adam', loss=losses.dsc, metrics=[metrics.dsc, metrics.iou])

        if summary:
            print(model.summary())
        return model
        pass


class VNet(FullyConvolutionalNetwork):
    def __init__(self, session_name, vol_size=(64, 64, 64), strides=(32, 32, 64), epochs=10, batch_size=32):
        super(VNet, self).__init__(session_name, vol_size, strides, epochs, batch_size)

    def model(self, summary=False, concat_axis=-1):
        """
        Builds a V-Net fully convolutional model
        :param summary: Show the model summary
        :param concat_axis: Axis for concatination layers
        :return: Compiled Keras model
        """
        # type: (int, bool) -> keras.models.Model
        vol_size = self.vol_size
        # 1_down
        inputs = Input(shape=(vol_size[0], vol_size[1], vol_size[2], 1))
        conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same', name='conv3d_0')(inputs)
        add1 = Add()([conv1, inputs])
        down_conv1 = Conv3D(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add1)

        # 2_down
        conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(down_conv1)
        conv2 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv2)
        add2 = Add()([conv2, down_conv1])
        down_conv2 = Conv3D(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add2)

        # 3_down
        conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(down_conv2)
        conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
        conv3 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv3)
        add3 = Add()([conv3, down_conv2])
        down_conv3 = Conv3D(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add3)

        # 4_down
        conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(down_conv3)
        conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
        conv4 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv4)
        add4 = Add()([conv4, down_conv3])
        down_conv4 = Conv3D(256, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add4)

        # 5
        conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(down_conv4)
        conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)
        conv5 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv5)
        add5 = Add()([conv5, down_conv4])
        up_conv1 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add5)

        # 4_up
        concat1 = Concatenate(axis=concat_axis)([add4, up_conv1])
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(concat1)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
        add6 = Add()([conv4, up_conv1])
        up_conv2 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add6)

        # 3_up
        concat2 = Concatenate(axis=concat_axis)([add3, up_conv2])
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(concat2)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
        add7 = Add()([conv3, up_conv2])
        up_conv3 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add7)

        # 2_up
        concat3 = Concatenate(axis=concat_axis)([add2, up_conv3])
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(concat3)
        conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
        add8 = Add()([conv2, up_conv3])
        up_conv4 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), activation='relu', padding='same')(add8)

        # 1_up
        concat4 = Concatenate(axis=concat_axis)([add1, up_conv4])
        conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(concat4)
        add9 = Add()([conv1, up_conv4])
        conv1 = Conv3D(1, (1, 1, 1), activation='softmax', padding='same')(add9)

        model = Model(inputs=inputs, outputs=conv1)
        model.compile(optimizer='adam', loss=losses.dsc, metrics=[metrics.dsc, metrics.iou])

        if summary:
            print(model.summary())
        return model
