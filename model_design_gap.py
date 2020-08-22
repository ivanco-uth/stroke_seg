from tensorflow.keras.layers import (Input, Conv3D, Conv2DTranspose,
                          MaxPool3D, Concatenate, UpSampling2D,
                          Activation, BatchNormalization, Dense, Flatten, Dropout,
                          GlobalAveragePooling3D)
import tensorflow as tf
from alt_model import create_unet_model3D


def save_model_if_val(best_val, current_val, model_path, model):

    out_val = None
    if current_val < best_val:
        print("Saving best model at {0}".format(model_path))
        model.save(filepath=model_path)
        out_val = current_val
    else:
        out_val = best_val

    return out_val


def model_def(input_image_size = (64, 64, 64, 1),
                        n_labels=1,
                        convolution_kernel_size=(3, 3, 3),
                        deconvolution_kernel_size=(3, 3, 3),
                        pool_size=(2, 2, 2),
                        mode='classification',
                        output_activation='sigmoid', data_seq = ["ct_1"], layers=2):

    print("Network Input Resolution: {0}".format(input_image_size))

    model_to_save = create_unet_model3D((input_image_size[0], input_image_size[1], input_image_size[2], len(data_seq)), n_labels=1, layers=2
                                                       , lowest_resolution=16, convolution_kernel_size=convolution_kernel_size
                                                       , deconvolution_kernel_size=deconvolution_kernel_size, pool_size=(2, 2, 2))

    model_to_save.summary()

    return model_to_save


def main():
    model_def()


if __name__ == '__main__':
    main()