# This code is imported from the following project: https://github.com/asmith26/wide_resnets_keras

import logging
import sys
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, add, Dense, Flatten, Dropout, Multiply, Embedding, Lambda, Add, Concatenate, Activation
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import SGD,Adam
from tensorflow.keras.applications import MobileNet
from densenet import *
from tensorflow.keras.utils import plot_model

sys.setrecursionlimit(2 ** 20)
np.random.seed(2 ** 10)


class TYY_MobileNet_reg:
    def __init__(self, image_size, alpha):
        
        if K.image_data_format() == 'channels_first':
        #if K.common.image_dim_ordering() == "th":
            logging.debug("image_data_format = 'channels_first'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_data_format = 'channels_last'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.alpha = alpha
        logging.debug("channel_axis={}".format(self._channel_axis))
        logging.debug("input_shape={}".format(self._input_shape))
        logging.debug("alpha={}".format(self.alpha))

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        logging.debug("input_shape={}".format(type(inputs)))
        #inputs = Input(shape=(64, 64, 3))
        model_mobilenet = MobileNet(input_shape=self._input_shape, alpha=self.alpha, depth_multiplier=1, dropout=1e-3, include_top=False, weights=None, input_tensor=None, pooling=None)
        x = model_mobilenet(inputs)
        #flatten = Flatten()(x)
        
        feat_a = Conv2D(20,(1,1),activation='relu')(x)
        feat_a = Flatten()(feat_a)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu',name='feat_a')(feat_a)

        pred_a = Dense(1,name='pred_a')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_a])


        return model


class TYY_DenseNet_reg:
    def __init__(self, image_size, depth):
        
        if K.common.image_dim_ordering() == "th":
            logging.debug("image_dim_ordering = 'th'")
            self._channel_axis = 1
            self._input_shape = (3, image_size, image_size)
        else:
            logging.debug("image_dim_ordering = 'tf'")
            self._channel_axis = -1
            self._input_shape = (image_size, image_size, 3)
        self.depth = depth

#    def create_model(self):
    def __call__(self):
        logging.debug("Creating model...")

        inputs = Input(shape=self._input_shape)
        model_densenet = DenseNet(input_shape=self._input_shape, depth=self.depth, include_top=False, weights=None, input_tensor=None)
        flatten = model_densenet(inputs)
        
        feat_a = Dense(128,activation='relu')(flatten)
        feat_a = Dropout(0.2)(feat_a)
        feat_a = Dense(32,activation='relu',name='feat_a')(feat_a)

        pred_a = Dense(1,name='pred_a')(feat_a)
        model = Model(inputs=inputs, outputs=[pred_a])
        
        return model


