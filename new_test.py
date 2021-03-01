# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 08:21:22 2021

@author: lenovo
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys
sys.path.append('D:\miniconda3\Lib\site-packages')
from Logger import log
from DataProvider import DoubleSourceProvider3
import NetFlowExt as nf
import nilm_metric as nm
from cnnModel import get_model, weights_loader
import os
import numpy as np
import tensorflow as tf
from keras.layers import Input
import pandas as pd
import argparse
from Arguments import *


def remove_space(string):
    return string.replace(" ","")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_arguments():
    parser = argparse.ArgumentParser(description='Predict the appliance\
                                     give a trained neural network\
                                     for energy disaggregation -\
                                     network input = mains window;\
                                     network target = the states of\
                                     the target appliance.')
    parser.add_argument('--appliance_name',
                        type=remove_space,
                        default='kettle',
                        help='the name of target appliance')
    parser.add_argument('--datadir',
                        type=str,
                        default='./dataset_management/refit/',
                        help='this is the directory to the test data')
    parser.add_argument('--trained_model_dir',
                        type=str,
                        default='./trained_model_new',
                        help='this is the directory to the trained models')
    parser.add_argument('--save_results_dir',
                        type=str,
                        default='./results_new',
                        help='this is the directory to save the predictions')
    parser.add_argument('--nosOfWindows',
                        type=int,
                        default=1000,
                        help='The number of windows for prediction \
                            for each iteration.')
    parser.add_argument('--dense_layers',
                        type=int,
                        default=1,
                        help=':\
                                1 -- One dense layers (default Seq2point);\
                                2 -- Two dense layers;\
                                3 -- three dense layers the CNN.')
    parser.add_argument("--transfer", type=str2bool,
                        default=False,
                        help="Using a pre-trained CNN (True) or not (False).")
    parser.add_argument("--plot_results", type=str2bool,
                        default=False,
                        help="To plot the predicted appliance against ground truth or not.")
    parser.add_argument('--cnn',
                        type=str,
                        default='kettle',
                        help='The trained CNN by which appliance to load.')
    parser.add_argument('--crop_dataset',
                        type=int,
                        default=None,
                        help='for debugging porpose should be helpful to crop the test dataset size')
    return parser.parse_args()


args = get_arguments()
log('Arguments: ')
log(args)


def load_dataset(filename, thr, header=0):
    data_frame = pd.read_csv(filename,
                             # nrows=args.crop_dataset,
                             header=header,
                             na_filter=False,
                             #memory_map=True
                             )

    test_set_x = np.round(np.array(data_frame.iloc[:, 0], float), 5)
    test_set_y = np.array(data_frame.iloc[:, 1])
    ground_truth = np.array(data_frame.iloc[offset:-offset, 1])
    del data_frame
    return test_set_x, test_set_y, ground_truth


appliance_name = args.appliance_name
log('Appliance target is: ' + appliance_name)

# Looking for the selected test set
for filename in os.listdir(args.datadir + appliance_name + '_class'):
        if 'TEST' in filename.upper() and 'TRAIN' not in filename.upper() and 'UK' not in filename.upper():
            test_filename = filename
            break

log('File for test: ' + test_filename)
loadname_test = args.datadir + appliance_name + '_class/' + test_filename
log('Loading from: ' + loadname_test)

# offset parameter from windowlenght
offset = int(0.5 * (params_appliance[args.appliance_name]['windowlength'] - 1.0))

test_set_x, test_set_y, ground_truth = load_dataset(loadname_test, params_appliance[args.appliance_name]['on_power_threshold'])
sess = tf.InteractiveSession()

# Dictonary containing the dataset input and target

test_kwag = {
    'inputs': test_set_x,
    'targets': test_set_y
}

# Defining object for training set loading and windowing provider
test_provider = DoubleSourceProvider3(nofWindows=args.nosOfWindows,
                                      offset=offset)

# TensorFlow placeholders
x = tf.placeholder(tf.float32,
                   shape=[None, params_appliance[args.appliance_name]['windowlength']],
                   name='x')

y_ = tf.placeholder(tf.float32,
                    shape=[None, 1],
                    name='y_')

# -------------------------------- Keras Network - from model.py -------------------------------------
inp = Input(tensor=x)

model = get_model(args.appliance_name,
                  inp,
                  params_appliance[args.appliance_name]['windowlength'],
                  n_dense=args.dense_layers
                  )[0]

y = model.outputs
# ----------------------------------------------------------------------------------------------------

sess.run(tf.global_variables_initializer())

# Load path depending on the model kind
if args.transfer:
    print('arg.transfer'.format(args.transfer))
    param_file = args.trained_model_dir+'/cnn_s2p_' + appliance_name + '_transf_' + args.cnn + '_pointnet_model'
else:
    print('arg.transfer'.format(args.transfer))
    param_file = args.trained_model_dir+'/cnn_s2p_' + args.appliance_name + '_pointnet_model'

# Loading weigths
log('Model file: {}'.format(param_file))
weights_loader(model, param_file)

# Calling custom test function
# test_prediction = nf.custompredictX(sess=sess,
#                                     network=model,
#                                     output_provider=test_provider,
#                                     x=x,
#                                     fragment_size=args.nosOfWindows,
#                                     output_length=1,
#                                     y_op=None,
#                                     out_kwag=test_kwag)
prediction = nf.custompredictX(sess=sess,
                               network=model,
                               output_provider=test_provider,
                               x=x,
                               fragment_size=args.nosOfWindows,
                               output_length=1,
                               y_op=None,
                               out_kwag=test_kwag)

prediction = np.array([int(item>0.5) for item in prediction])

# ------------------------------------- Performance evaluation----------------------------------------------------------

# Parameters
max_power = params_appliance[args.appliance_name]['max_on_power']
threshold = params_appliance[args.appliance_name]['on_power_threshold']
aggregate_mean = 522
aggregate_std = 814


log('aggregate_mean: ' + str(aggregate_mean))
log('aggregate_std: ' + str(aggregate_std))


sess.close()


# ------------------------------------------ metric evaluation----------------------------------------------------------
sample_second = 8.0  # sample time is 8 seconds
log('F1:{0}'.format(nm.get_F1(ground_truth.flatten(), prediction.flatten(), threshold)))


# ----------------------------------------------- save results ---------------------------------------------------------
savemains = test_set_x.flatten() * aggregate_std + aggregate_mean
savegt = ground_truth
savepred = prediction.flatten()

if args.transfer:
    save_name = args.save_results_dir + '/' + appliance_name  # save path for mains
else:
    save_name = args.save_results_dir + '/' + appliance_name   # save path for mains
if not os.path.exists(save_name):
        os.makedirs(save_name)
# Numpy saving
np.save(save_name + '/pred.npy', savepred)
np.save(save_name + '/gt.npy', savegt)
np.save(save_name + '/mains.npy', savemains)


"""
# saving in csv format
result_dict = {
    'aggregate': savepred,
    'ground truth': savegt,
    'prediction': savepred,
}

# CSV saving
result = pd.DataFrame(result_dict)
result.to_csv(save_name + '.csv', index=False)
"""

log('size: x={0}, y={0}, gt={0}'.format(np.shape(savemains), np.shape(savepred), np.shape(savegt)))
# ----------------------------------------------- PLOT results ---------------------------------------------------------
if args.plot_results:

    import matplotlib.pyplot as plt

    if args.plot_results:
        offset = int(0.5*198)
        fig1 = plt.figure()
        random_num = np.random.randint(0,len(savegt)-1500)

        ax1 = fig1.add_subplot(111)
        ax1.plot(ground_truth[random_num:random_num+1500], color='#d62728', linewidth=1.6)
        ax1.plot(prediction[random_num:random_num+1500],
                 color='#1f77b4',
                 #marker='o',
                 linewidth=1.5)
        ax1.grid()
        ax1.set_ylabel('W')
        ax1.legend(['ground truth', 'prediction'])
        mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        plt.show(fig1)
