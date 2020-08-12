import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, Conv2DTranspose,
                          MaxPooling2D, Concatenate, UpSampling2D,
                          Activation, BatchNormalization)
from tensorflow.keras.layers import (Conv3D, Conv3DTranspose,
                          MaxPooling3D,  UpSampling3D)

from tensorflow.keras import optimizers as opt
from itertools import product
from aux_metrics import *
from tensorflow.keras.utils import multi_gpu_model


def create_unet_model3D(input_image_size,
                        n_labels=1,
                        layers=4,
                        lowest_resolution=16,
                        convolution_kernel_size=(5, 5),
                        deconvolution_kernel_size=(5, 5),
                        pool_size=(2, 2),
                        mode='classification',
                        output_activation='tanh',
                        init_lr=0.0001, class_weights = {}, gpu_num = 2):


    # Turns number of layers into iterable list
    layers = np.arange(layers)

    # Sets number of classification labels (number of distinct tissues to be considered)
    number_of_classification_labels = n_labels

    # Defines input layer to model
    inputs = Input(shape=input_image_size)

    # defines enconding section of the network, assigns variables amount of convolutional layers depending
    encoding_convolution_layers = []
    pool = None

    for i in range(len(layers)):
        number_of_filters = lowest_resolution * 2 ** (layers[i])

        if i == 0:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                           padding='same')(inputs)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)


        elif i == 1:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          padding='same')(pool)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

            # conv = Conv3D(filters=number_of_filters,
            #               kernel_size=convolution_kernel_size,
            #               padding='same')(pool)
            # conv = BatchNormalization()(conv)
            # conv = Activation('relu')(conv)


        else:
            conv = Conv3D(filters=number_of_filters,
                          kernel_size=convolution_kernel_size,
                          padding='same')(pool)
            conv = BatchNormalization()(conv)
            conv = Activation('relu')(conv)

            # conv = Conv3D(filters=number_of_filters,
            #               kernel_size=convolution_kernel_size,
            #               padding='same')(conv)
            # conv = BatchNormalization()(conv)
            # conv = Activation('relu')(conv)



        conv_buff = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                           padding='same')(conv)
        conv_buff = BatchNormalization()(conv_buff)
        conv_buff = Activation('relu')(conv_buff)


        encoding_convolution_layers.append(conv_buff)


        if i < len(layers) - 1:
            pool = MaxPooling3D(pool_size=pool_size)(encoding_convolution_layers[i])

    ## DECODING PATH ##
    outputs = encoding_convolution_layers[len(layers) - 1]
    for i in range(1, len(layers)):
        number_of_filters = lowest_resolution * 2 ** (len(layers) - layers[i] - 1)
        
        tmp_deconv = Conv3DTranspose(filters=number_of_filters, kernel_size=deconvolution_kernel_size,
                                     padding='same')(outputs)

        tmp_deconv = UpSampling3D(size=pool_size)(tmp_deconv)
        outputs = Concatenate(axis=-1)([tmp_deconv, encoding_convolution_layers[len(layers) - i - 1]])


        if i == 1 or i == 2:
            outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                          padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

            outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

            # outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size, padding='same')(outputs)
            # outputs = BatchNormalization()(outputs)
            # outputs = Activation('relu')(outputs)

        else:
            outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size,
                          padding='same')(outputs)
            outputs = BatchNormalization()(outputs)
            outputs = Activation('relu')(outputs)

            # outputs = Conv3D(filters=number_of_filters, kernel_size=convolution_kernel_size,
            #                  padding='same')(outputs)
            # outputs = BatchNormalization()(outputs)
            # outputs = Activation('relu')(outputs)


    # network is used for classification which requires sigmoid activation for last layer if only tissue is classified against background
    # last layer is set to softmax if there is more than one tissue to be classified
    if mode == 'classification':


        if number_of_classification_labels == 1:
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1, 1, 1),
                             activation='sigmoid')(outputs)
        else:
            
            outputs = Conv3D(filters=number_of_classification_labels, kernel_size=(1, 1, 1),
                             activation='softmax')(outputs)

        # Joins previously defined layers of model with newly defined output layer
        unet_model = Model(inputs=inputs, outputs=outputs)


    return unet_model