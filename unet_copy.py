import os
import sys
import time
import random
import shutil
import datetime
import numpy as np
import nibabel as nib
import tensorflow as tf
from itertools import cycle
from matplotlib import pyplot as plt
from data_manipulation import get_data_partitions, populate_data_array, transform_data
from model_design_gap import model_def, save_model_if_val
from sklearn.metrics import f1_score, confusion_matrix 
from tensorflow.python.keras import backend as K
from process_data_ich import get_cases
from sklearn.model_selection import train_test_split as data_split
from scipy.ndimage import zoom
from tensorflow.keras.losses import BinaryCrossentropy
from mri_functions import save_obj, load_obj
from scipy.ndimage import rotate
from multiprocessing import Process
from math import ceil
from net_architecture import i_model


# tf.keras.backend.set_floatx('float16')

os.environ["CUDA_VISIBLE_DEVICES"] = "14"
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
tf.config.experimental.set_memory_growth(gpus[0] , True)
# tf.config.run_functions_eagerly(True)

# run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

def process_image(img_data, out_resolution, norm_limits=[0,100], normalizing=False):

    # print("Processing image")

    if normalizing:

        img_data[img_data < norm_limits[0]] = norm_limits[0]
        img_data[img_data > norm_limits[1]] = norm_limits[1]

        img_data = (img_data - np.min(img_data))/(np.max(img_data) - np.min(img_data))

    else:

        img_data[img_data != 0] = 1
        
        # img[img != 0] = 1

    return img_data


def volume_crop(img_data=None, target_resolution=None
        , random_crop= False, target_coords=None, step_in=None):

    # print("In Image Shape: {0}".format(img_data.shape))

    x_step, y_step, z_step = target_resolution[0], target_resolution[1], target_resolution[2]

    x_limit, y_limit, z_limit = img_data.shape[0], img_data.shape[1], img_data.shape[2]

    crop_data = np.zeros((x_step, y_step, z_step))
    crop_coords = None


    if random_crop:
        x_coord, y_coord, z_coord = random.randint(0 , x_limit), random.randint(0 , y_limit), random.randint(0 , z_limit)
        # print("Random Coords: ({0}, {1}, {2})".format(x_coord, y_coord, z_coord))
        
        if x_coord + x_step > x_limit:
            x_coord = x_limit - x_step
        if y_coord + y_step > y_limit:
            y_coord = y_limit - y_step
        if z_coord + z_step > z_limit:
            z_coord = z_limit - z_step 

        crop_data[:, :, :] = img_data[x_coord: x_coord + x_step, y_coord:y_coord + y_step, z_coord:z_coord + z_step]
        crop_coords = [x_coord, y_coord, z_coord]
        # print("Crop Shape: {0}".format(crop_data.shape))

    elif not random_crop and step_in is None:

        x_coord, y_coord, z_coord = target_coords[0], target_coords[1], target_coords[2]

        crop_data[:, :, :] = img_data[x_coord: x_coord + x_step, y_coord:y_coord + y_step, z_coord:z_coord + z_step]
        crop_coords = [x_coord, y_coord, z_coord]
             
    return crop_data, crop_coords

def load_process_stroke_file(data_list, data_dict, batch_size, 
file_path, label_file_path, out_resolution, new_file_path, new_label_file_path,
 save_new, buffer_size, augment_data):

    batch_count = 0

    data_array = np.zeros((buffer_size, out_resolution[0]
    , out_resolution[1], out_resolution[2], 1))

    label_array = np.zeros((buffer_size, out_resolution[0]
    , out_resolution[1], out_resolution[2], 1))

    while batch_count < buffer_size:

        # print("At while loop")

        data_id = next(data_list)

        # print("Current case ID: {0}".format(data_id))

        img_label_check = False

        if data_dict[data_id] == 1:
            img_label_check = True

        old_path = file_path.format(data_id)
        new_path = new_file_path.format(data_id)

        # print("ID: {0} : Label {1}".format(data_id, data_dict[data_id]))


        old_label_path = label_file_path.format(data_id)
        new_label_path = new_label_file_path.format(data_id)

        # print(old_label_path)
        # print(new_label_path)

        new_label_data = None


        # print("Before loading: {0}".format(save_new))

        if save_new:
            
            try:
                
                img_file = nib.load(filename=old_path)
                img_data = img_file.get_fdata()

                new_img_data = zoom(input=img_data, zoom= (float(out_resolution[0]/img_data.shape[0]), 
                float(out_resolution[1]/img_data.shape[1]), float(out_resolution[2]/img_data.shape[2])), mode='nearest' , prefilter=True)

                new_img_data = process_image(img_data=new_img_data
                , out_resolution=out_resolution, normalizing=True)

                np.save(file=new_path, arr=new_img_data)

                if img_label_check:

                    label_file = nib.load(filename=old_label_path)
                    label_data = label_file.get_fdata()

                    new_label = zoom(input=label_data, zoom= (float(out_resolution[0]/label_data.shape[0]), 
                    float(out_resolution[1]/label_data.shape[1]), float(out_resolution[2]/label_data.shape[2])), mode='nearest' , prefilter=False)

                    # print("Before pre_proc label: {0}".format(np.unique(new_label, return_counts=True)))

                    new_label_data = process_image(img_data= new_label
                    , out_resolution= out_resolution, normalizing=False)

                    # print("After pre-proc label: {0}".format(np.unique(new_label_data, return_counts=True)))

                    np.save(file=new_label_path, arr=new_label_data)

                else:
                    new_label_data = np.zeros((new_img_data.shape[0]
                    , new_img_data.shape[1], new_img_data.shape[2]))

            except:
                print("File {0} not found".format(new_label_file_path.format(data_id)))
                continue

        else:
            try:

                new_img_data = np.load(file=new_path)
              
                if img_label_check:
                    new_label_data = np.load(file=new_label_path)
                
                else:
                    new_label_data = np.zeros((new_img_data.shape[0]
                    , new_img_data.shape[1], new_img_data.shape[2]))
                                   
            except FileNotFoundError:
                # print("File not found error | path: {0}".format(new_path))
                continue

            except:
                raise

        crop_data = new_img_data
        crop_label = new_label_data


        rand_x_angle = random.randint(-5, 5)
        rand_y_angle = random.randint(-5, 5)
        rand_z_angle = random.randint(-5, 5)

        rand_x_rot = random.randint(0, 1)
        rand_y_rot = random.randint(0, 1)
        rand_z_rot = random.randint(0, 1)

        if augment_data == True:

            if rand_x_rot:
                new_img_data = rotate(input= crop_data, angle=rand_x_angle,
                axes=(0, 1,), reshape=False)

                new_label_data = rotate(input= crop_label, angle=rand_x_angle,
                axes=(0, 1,), reshape=False)

            if rand_y_rot:
                new_img_data = rotate(input= crop_data, angle=rand_y_angle,
                axes=(1, 2,), reshape=False)

                new_label_data = rotate(input= crop_label, angle=rand_y_angle,
                axes=(1, 2,), reshape=False)

            if rand_z_rot:
                new_img_data = rotate(input= crop_data, angle=rand_z_angle,
                axes=(0, 2,), reshape=False)

                new_label_data = rotate(input= crop_label, angle=rand_z_angle,
                axes=(0, 2,), reshape=False)

        data_array[batch_count, :, :, :, 0]= crop_data

        label_array[batch_count, :, :, :, 0] = crop_label
        batch_count += 1


    train_batch = tf.data.Dataset.from_tensor_slices((data_array.astype('float32'), label_array.astype('float32')))
    train_batch = train_batch.batch(batch_size)    

    return train_batch



def train_wrapper(model, loss_fn, opti):
    @tf.function
    def training_step(x, y,  train_auc_metric=None
                        , train_acc_metric=None, train_tp_metric=None,
                        train_fp_metric=None, train_fn_metric=None, train_tn_metric=None):

        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            loss_value = loss_fn(y, logits)

        grads = tape.gradient(loss_value, model.trainable_weights)
        opti.apply_gradients(zip(grads, model.trainable_weights))
        
        # train_acc_metric.update_state(y, logits)
        # train_tp_metric.update_state(y, logits)
        # train_fn_metric.update_state(y, logits)
        # train_fp_metric.update_state(y, logits)
        # train_tn_metric.update_state(y, logits)
        # train_auc_metric.update_state(y, logits)

        return loss_value
    return training_step


def validation_wrapper(model, loss_fn):
    @tf.function
    def validation_step(x, y,  val_auc_metric=None
                        , val_acc_metric=None, val_tp_metric=None,
                        val_fp_metric=None, val_fn_metric=None, val_tn_metric=None):

        logits = model(x, training=False)
        loss_value = loss_fn(y, logits)
        
        # val_acc_metric.update_state(y, logits)
        # val_tp_metric.update_state(y, logits)
        # val_fn_metric.update_state(y, logits)
        # val_fp_metric.update_state(y, logits)
        # val_tn_metric.update_state(y, logits)
        # val_auc_metric.update_state(y, logits)

        return loss_value
    return validation_step



def conduct_training(batch_size, lr_rate, loss_func, data_folder, label_folder, epochs, op_mode):

    print("Currently doing {0}".format(op_mode))

    # get data plus labels

    file_folder = "/collab/gianca-group/icoronado/ct_stroke/preproc_data/"

    file_path = "/collab/gianca-group/icoronado/ct_stroke/preproc_data/{0}-ct_brain_reg.nii.gz"

    new_file_path = "/collab/gianca-group/icoronado/stroke_files/preproc_brain/{0}-ct_brain_reg.npy"

    new_file_folder = "/collab/gianca-group/icoronado/stroke_files/preproc_brain/"



    label_file_path = "/collab/gianca-group/icoronado/stroke_files/mask_out/{0}/{0}-mask_fov_flirt.nii.gz"

    new_label_file_path = "/collab/gianca-group/icoronado/stroke_files/preproc_label/{0}-mask_fov_flirt.npy"

    save_new_files = False
    save_new_files_batch = False

    if save_new_files:

        try:
            shutil.rmtree(path=new_file_folder)
            os.mkdir(path=new_file_folder)
        except FileNotFoundError:
            os.mkdir(path=new_file_folder)

    # sys.exit()

    file_info_list = get_cases()

    pos_cases, neg_cases = [], []

    for key in file_info_list:
        pair = file_info_list[key]
        # print("Key: {0} -> {1}".format(key, pair))
        
        if int(pair) > 0:
            pos_cases.append(key)
        else:
            neg_cases.append(key)

    
    # folders to be redefined
    # train_cases, val_cases, test_cases = get_data_partitions(train_part=0.7, val_part=0.1, data_folder=data_folder, random_seed=0)


    random_state = 15

    train_pos, test_pos = data_split(pos_cases, test_size=0.1, train_size=0.6, shuffle=True, random_state=random_state)
    train_neg, test_neg = data_split(neg_cases, test_size=0.1, train_size=0.6, shuffle=True, random_state=random_state)
    
    val_pos = []
    val_neg = []

    for case in pos_cases:
        if case not in train_pos and case not in test_pos:
            val_pos.append(case)

    for case in neg_cases:
        if case not in train_neg and case not in test_neg:
            val_neg.append(case)     


    print(len(pos_cases))
    print(len(neg_cases))

    print("Number training: {0}".format(len(train_pos + train_neg)))
    print("Number validation: {0}".format(len(val_neg + val_pos)))
    print("Number testing: {0}".format(len(test_pos + test_neg)))

    test_set = test_pos + test_neg

    save_obj(obj=test_set, name="test_data", obj_folder=".")

    for test_case in test_set:

        print(file_info_list[test_case])


    # optimizer = opt

    loss_fn = loss_func

    img_resolution, data_seq = (364, 436, 364, 1), ["t1_pre"]
    target_resolution = (128, 128, 128, 1)


    # model = model_def(input_image_size=target_resolution)


    model = i_model()

    # sys.exit()





    model_path = "model_res.hdf5"


    # model.load_weights("model_baseline_ich_n.hdf5")

    
    buffer_size = batch_size * 1


    val_batch_size = batch_size * 1
    val_buffer_size = batch_size * 1
    


    if op_mode == 'train':


        metrics_list = np.zeros((epochs, 5))

        train_acc_metric = tf.keras.metrics.BinaryAccuracy()
        train_tp_metric = tf.keras.metrics.TruePositives()
        train_tn_metric = tf.keras.metrics.TrueNegatives()
        train_fp_metric = tf.keras.metrics.FalsePositives()
        train_fn_metric = tf.keras.metrics.FalseNegatives()
        train_auc_metric = tf.keras.metrics.AUC()

        train_metrics = [train_acc_metric, train_tp_metric, train_tn_metric,
        train_fp_metric, train_fn_metric, train_auc_metric]


        val_acc_metric = tf.keras.metrics.BinaryAccuracy()
        val_tp_metric = tf.keras.metrics.TruePositives()
        val_tn_metric = tf.keras.metrics.TrueNegatives()
        val_fp_metric = tf.keras.metrics.FalsePositives()
        val_fn_metric = tf.keras.metrics.FalseNegatives()
        val_auc_metric = tf.keras.metrics.AUC()


        val_metrics = [val_acc_metric, val_auc_metric, val_tp_metric,
        val_tn_metric, val_fn_metric, val_fp_metric]

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        global curr_epoch

        best_loss = 1

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)

        train_set = train_pos + train_neg
        val_set = val_pos + val_neg

        print(train_set)
        print(val_set)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate, amsgrad=True)

        train_one_batch = train_wrapper(model=model, loss_fn=loss_fn, opti=optimizer)

        val_one_batch = validation_wrapper(model=model, loss_fn=loss_fn)

        
       
        for epoch in range(epochs):

            start = time.time()

            

            if epoch > 0:
                save_new_files = False
            
            print('Start of epoch %d' % (epoch,))
            random.Random(5).shuffle(train_set)
            train_iterator = cycle(train_set)

            # load image into tensorflow batch object
            # go through images process them and save them

            random.Random(5).shuffle(val_set)
            val_iterator = cycle(val_set)
            

            for step in range(ceil(len(train_set)/buffer_size)):

                if save_new_files:
                    print("Saved {0} sets of images out of {1}".format(step, len(train_set)))

                # print("Here 1")

                # print("BatchSize: {0} | Data Seen: {1} | Total Data: {2}".format(batch_size, step*buffer_size, len(train_set)))

                dataset_batch = load_process_stroke_file(data_list=train_iterator, 
                data_dict= file_info_list, batch_size=batch_size, file_path=file_path, 
                label_file_path=label_file_path, new_label_file_path=new_label_file_path
                , out_resolution=target_resolution, new_file_path=new_file_path
                , save_new=save_new_files, buffer_size=buffer_size, augment_data=True)


                # print(dataset_batch)

                # print("Dataset Batch")

                # sys.exit()

                for batch_step, (x_batch_train, y_batch_train) in enumerate(dataset_batch):

                    # print(x_batch_train.shape)
                    # print(np.unique(x_batch_train))
                    # print(y_batch_train.shape)
                    # print(np.unique(y_batch_train))

                    

                    loss_value = train_one_batch(x=x_batch_train, y=y_batch_train
                    , train_auc_metric=train_auc_metric
                    , train_acc_metric=train_acc_metric, train_tp_metric=train_tp_metric,
                    train_fp_metric=train_fp_metric, train_fn_metric=train_fn_metric, train_tn_metric=train_tn_metric)


                    train_loss(loss_value)

                    # tf.keras.backend.clear_session()

            # print("Finish Training")

            # train_auc = train_auc_metric.result()
            # train_acc = train_acc_metric.result()

            # train_fp = train_fp_metric.result()
            # train_fn = train_fn_metric.result()
            # train_tp = train_tp_metric.result()

            # train_f1 = (2 * train_tp)/(2*train_tp + train_fp + train_fn)

            # print("Training F1 Score: {0}".format(train_f1))

            print("Train Loss: {0}".format(train_loss.result()))
            
            
            # print("Loss: {0} | AUC: {1} |  Acc: {2}".format(train_loss.result(), train_auc, train_acc))



            val_dataset = load_process_stroke_file(data_list=val_iterator, 
                data_dict= file_info_list, batch_size=val_batch_size, file_path=file_path
                , out_resolution=target_resolution, new_file_path=new_file_path,
                label_file_path=label_file_path, new_label_file_path=new_label_file_path
                , save_new=save_new_files_batch, buffer_size=len(val_set), augment_data=False)

            if epoch > 0:
                save_new_files_batch = False


            for x_batch_val, y_batch_val in val_dataset:

                
                val_loss_value = val_one_batch(x=x_batch_val, y=y_batch_val
                    , val_auc_metric=val_auc_metric
                    , val_acc_metric=val_acc_metric, val_tp_metric=val_tp_metric,
                    val_fp_metric=val_fp_metric, val_fn_metric=val_fn_metric, val_tn_metric=val_tn_metric)
                # Update val metrics
                

                val_loss(val_loss_value)


            FP_V = 0
            TP_V = 0 
            FN_V = 0 

            if epoch % 20 == 0:

                for idx_batch , (x_batch, y_batch) in enumerate(val_dataset):

                    result = model.predict(x_batch)

                    # print("Unique Val Predict: {0}".format(np.unique(result)))

                    result = np.nan_to_num(result)

                    result[result >= 0.5] = 1
                    result[result < 0.5] = 0

                    
            
                    label_ground_truth = y_batch.numpy()
                    label_predicted = result


                    print("Ground Truth Unique: {0}".format(np.unique(label_ground_truth, return_counts=True)))
                    print("Predicted Unique: {0}".format(np.unique(label_predicted, return_counts=True)))


                    con_max = confusion_matrix(y_true=label_ground_truth.flatten() 
                    , y_pred= label_predicted.flatten())

                    FP = con_max[0][1]
                    TP = con_max[1][1]
                    FN = con_max[1][0]

                    FP_V += FP
                    TP_V += TP
                    FN_V += FN


                dsc_val = (2*TP_V)/(2*TP_V + FP_V + FN_V)

                print("Validation F1: {0}".format(dsc_val))

            # sys.exit()





            
            # val_acc = val_acc_metric.result()
            # val_auc = val_auc_metric.result()
            # val_tp = val_tp_metric.result()
            # val_fp = val_fp_metric.result()
            # val_fn = val_fn_metric.result() 

            # print('Validation Loss: {0} | AUC: {1} | Acc: {2}'.format(val_loss.result()
            # , val_auc, val_acc))

            print('Validation Loss: {0}'.format(val_loss.result()))

            best_auc = save_model_if_val(best_loss, val_loss.result(), model_path, model)

            
            # val_fp = val_fp_metric.result()
            # val_fn = val_fn_metric.result()
            # val_tp = val_tp_metric.result()

            # val_f1 = (2 * val_tp)/(2*val_tp + val_fp + val_fn)

            # print("Validation F1 Score: {0}".format(val_f1))


            # with train_summary_writer.as_default():
            #     tf.summary.scalar('Training loss', train_loss.result(), step=epoch)
            #     tf.summary.scalar('Training AUC', train_auc, step=epoch)
            #     tf.summary.scalar('Training accuracy', train_acc, step=epoch)
            #     tf.summary.scalar('val loss', val_loss.result(), step=epoch)
            #     tf.summary.scalar('Val AUC', val_auc, step=epoch)
            #     tf.summary.scalar('Val accuracy', val_acc, step=epoch)
            #     tf.summary.scalar("Training DSC", train_f1, step=epoch)
            #     tf.summary.scalar("Val DSC", val_f1, step=epoch)


            # train_auc_metric.reset_states()
            # train_fn_metric.reset_states()
            # train_acc_metric.reset_states()
            # train_fp_metric.reset_states()
            # train_tn_metric.reset_states()
            # train_tp_metric.reset_states()
            
            # val_acc_metric.reset_states()

            end = time.time()

            print("Time elapsed: {:.2f}".format((end-start)/60))


            print(":-) :-) :-) (-: (-: (-:")

            # print("Here")
                




    elif op_mode == "test":


        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_rate, amsgrad=True)

        test_predicted = np.zeros((len(test_set)
        , target_resolution[0], target_resolution[1], target_resolution[2], 1))

        test_label = np.zeros((len(test_set), target_resolution[0]
        , target_resolution[1], target_resolution[2], 1))
        

        print("Loaded model for testing")
        model.load_weights(model_path)
        model.compile(optimizer=optimizer, loss=loss_fn)

        test_iterator = cycle(test_set)
        clone_iterator = cycle(test_set)

        save_new_files = True

        buffer_size = 1


        TN_T, FP_T, TP_T, FN_T = 0, 0, 0, 0


        for step in range(int(len(test_set)/buffer_size)):
            
            dataset_batch = load_process_stroke_file(data_list=test_iterator, 
            data_dict= file_info_list, batch_size=batch_size, file_path=file_path
            , out_resolution=target_resolution, new_file_path=new_file_path,
            label_file_path=label_file_path, new_label_file_path=new_label_file_path
            , save_new=save_new_files, buffer_size=buffer_size, augment_data=False)

            # print("Current Test Case ID: {0}".format(next(clone_iterator)))

            for batch_step, (x_batch_test, y_batch_test) in enumerate(dataset_batch):

                result = model.predict(x_batch_test)

                print("Predicted shape: {0}".format(result.shape))
                
                print("Label shape: {0}".format(y_batch_test.shape))

                for batch_i in range(len(result)):
                    pred_val = None

                    result[result >= 0.5] = 1
                    result[result < 0.5] = 0

                    np.save(file= "predicted_out/{0}.npy".format(next(clone_iterator)), arr=result[0, :, :, :, 0])

                    
                    label_ground_truth = y_batch_test.numpy()
                    label_predicted = result


                    print("Ground Truth Unique: {0}".format(np.unique(label_ground_truth, return_counts=True)))
                    print("Predicted Unique: {0}".format(np.unique(label_predicted, return_counts=True)))


                    

                    con_max = confusion_matrix(y_true=label_ground_truth.flatten() 
                    , y_pred= label_predicted.flatten())

                    print(con_max)

                    if len(con_max) > 1:

                        TN = con_max[0][0]
                        FP = con_max[0][1]
                        TP = con_max[1][1]
                        FN = con_max[1][0]

                        TN_T += TN
                        FP_T += FP
                        TP_T += TP
                        FN_T += FN

                        print("TN: {0} | FP: {1} | TP: {2} | FN: {3}".format(TN, FP, TP, FN))


                    # test_label[batch_i, :, :, :, 0] = y_batch_test[batch_i, :, :, :, 0]

                    # test_predicted[batch_i, :, :, :, 0] = result[batch_i, :, :, :, 0]


        print(test_predicted.shape)
        print(test_label.shape)      

        print("TN: {0} | FP: {1} | TP: {2} | FN: {3}".format(TN_T, FP_T, TP_T, FN_T))

        dsc_all = (2*TP_T)/(2*TP_T + FP_T + FN_T)

        print("Overall F1: {0}".format(dsc_all))

        


def main():
    # specify variables for training    
    epochs = 2000
    op_mode = 'train'
    lr_rate, batch_size = 1e-5, 4
    data_folder, label_folder = "../../brains", "../../lesions"
    loss_func = tf.losses.BinaryCrossentropy(from_logits=True)
    

    conduct_training(batch_size=batch_size, lr_rate=lr_rate, loss_func=loss_func,
    data_folder=data_folder, label_folder=label_folder, epochs=epochs, op_mode=op_mode)


if __name__ == '__main__':
    main()