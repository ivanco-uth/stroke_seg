import os
import sys
import cv2
import numpy as np
import random
from math import floor
import tensorflow as tf
from matplotlib import pyplot as plt

def transform_data(img_batch = None, label_batch = None, tgt_re = (256, 256), rotation = True, reflection = True, resize = True):

    new_sample = np.zeros((img_batch.shape[0], tgt_re[0], tgt_re[1], img_batch.shape[-1]))
    new_label =  np.zeros((label_batch.shape[0], tgt_re[0], tgt_re[1], label_batch.shape[-1]))

    for i in range(img_batch.shape[0]):

        

        mirror_x, mirror_y = random.randint(0, 1), random.randint(0, 1)
        rot_by_90 = random.randint(0, 1)
        ref_rot_angle = random.randint(1, 3)

        current_sample = img_batch[i, :, :, :]
        current_label = label_batch[i, :, :, :]

        

        # current_sample = tf.keras.preprocessing.image.img_to_array(current_sample)
        

        # if resize:

        #     current_sample = tf.image.resize(current_sample, size=tgt_re, method='nearest')
        #     current_sample = cv2.resize(current_sample, dsize=tgt_re, interpolation=cv2.INTER_NEAREST)
        #     current_sample = np.expand_dims(current_sample, axis=-1)


        # current_sample = current_sample
        # current_label = current_label

        if mirror_x and reflection:
            current_sample = tf.image.flip_left_right(current_sample)
            current_label = tf.image.flip_left_right(current_label)

        if mirror_y and reflection:
            current_sample = tf.image.flip_up_down(current_sample)
            current_label = tf.image.flip_up_down(current_label)
        if rot_by_90 and rotation:
            for a in range(ref_rot_angle):
                current_sample = tf.image.rot90(current_sample)
                current_label = tf.image.rot90(current_label)
                
        new_sample[i, :, :, :] = current_sample
        new_label[i, :, :, :] = current_label
        
    return new_sample, new_label    


# function to partition data into patient cases
def get_data_partitions(train_part = 0.6, val_part = 0.2, random_seed = 0, data_folder = "../brains"):

    # prints files in data folder
    data_files = os.listdir(data_folder)
    data_files.sort()
    # print(len(data_files))

    # contains cases for all files (not unique)
    cases_list_raw = [case_name.split("_")[0] for case_name in data_files]  

    # gets unique cases from cases list
    cases_list_raw = list(set(cases_list_raw))
    # print("Unique cases: {0}".format(len(cases_list_raw)))

    # shuffles the list randomly according to seed
    cases_list_raw.sort()
    random.Random(random_seed).shuffle(cases_list_raw)

    # print(cases_list_raw)
    
    # partitions data into train and validation sets
    train_cases, val_cases = cases_list_raw[0: int(len(cases_list_raw)*train_part)], cases_list_raw[int(len(cases_list_raw)*train_part):int(len(cases_list_raw)*(train_part + val_part))]

    # test set from data 
    test_cases = np.setdiff1d(cases_list_raw, train_cases + val_cases)

    return train_cases, val_cases, test_cases
    

# function to get files of slices according to patient case
def relate_files_cases(data_folder, cases_lists):

    files_list = []
    # iterate through cases
    for case_id in cases_lists:
        # iterate through list of slice files
        for slice_file in os.listdir("{0}".format(data_folder)): 
            # check whether file corresponds to a patient case
            if case_id in slice_file.split("_")[0]:
                files_list.append(slice_file)

    return files_list



def populate_data_array(data_folder, label_folder, cases_lists, num_sequences=1):

    
    # get files corresponding to patients
    slice_files_list = relate_files_cases(data_folder=data_folder, cases_lists=cases_lists)

    # load a sample image for configuring array
    sample_img = np.load("{0}/{1}".format(data_folder, slice_files_list[0]))
    sample_img_shape = sample_img.shape

    # arrays to hold data and labels
    data_array = np.zeros((len(slice_files_list), sample_img_shape[0], sample_img_shape[1], num_sequences))
    label_array = np.zeros((len(slice_files_list), sample_img_shape[0], sample_img_shape[1], 1 ))

    # iterate through slices files 
    for slice_idx, slice_file in enumerate(slice_files_list):
        
        # load data and labels
        slice_array = np.load("{0}/{1}".format(data_folder, slice_file))
        slice_label = np.load("{0}/{1}".format(label_folder, slice_file))

        # define label according to ground truth segmentation
        # if np.any(slice_label):
        #     slice_label = 1
        # else:
        #     slice_label = 0

        # save sample images and labels into arrays
        data_array[slice_idx, :, :, 0] = slice_array

        # print(slice_label.shape)
        label_array[slice_idx, :, :,  0] = slice_label

        label_array = np.where(label_array > 0, 1, 0)

    return data_array, label_array


def main():

    # Test for data loading

    data_folder, label_folder = "../brains", "../lesions"
    train_cases, val_cases, test_cases = get_data_partitions(train_part=0.7, val_part=0.1, data_folder="../brains", random_seed=0)

    case_id = train_cases[0]
    print("First case ID: {0}".format(case_id))

    # Test for populating array with data

    data, label = populate_data_array(data_folder=data_folder , label_folder=label_folder, cases_lists=train_cases)

    # plt.subplot(2, 1, 1)
    # plt.imshow(data[0,:,:,0])
    # plt.subplot(2, 1, 2)
    # plt.imshow(label[0,:,:,0])
    # plt.show()

    
    # Test for image manipulation

    # sample_img = np.load("{0}/{1}".format(data_folder, train_cases[0]))


    # sample_img = np.expand_dims(sample_img, axis=0)
    # sample_img = np.expand_dims(sample_img, axis=-1)


    # transformed_img = transform_data(img_batch=sample_img)

    # plt.subplot(2,1,1)
    # plt.imshow(sample_img[0, :, :, 0])
    # plt.subplot(2,1,2)
    # plt.imshow(transformed_img[0, :, :, 0])
    # plt.show()
    


if __name__ == '__main__':
    main()



