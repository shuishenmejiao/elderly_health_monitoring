# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 13:28:26 2021

@author: lenovo
"""
import pandas as pd
import os
from imblearn.under_sampling import RandomUnderSampler 


aggregate_mean = 522
aggregate_std = 814
params_appliance = {
    'kettle': {
        'windowlength': 199,
        'on_power_threshold': 2000,
        'max_on_power': 3998,
        'mean': 700,
        'std': 1000,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 19, 20],
        'channels': [8, 9, 9, 8, 7, 9, 9, 7, 6, 9, 5, 9],
        'train_house': [3, 4, 5, 6, 7, 8, 9, 12],
        'test_house': [2],
        'validation_house': [13, 19, 20]
    },
    'microwave': {
        'windowlength': 199,
        'on_power_threshold': 200,
        'max_on_power': 3969,
        'mean': 500,
        'std': 800,
        's2s_length': 128,
        'houses': [2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 15, 17, 18, 19, 20],
        'channels': [5, 8, 8, 7, 6, 8, 6, 8, 3, 8, 7, 7, 9, 4, 8],
        'train_house': [2, 3, 5, 6, 8, 9, 10, 12, 13, 15],
        'test_house': [4],
        'validation_house': [17, 18, 19, 20]
    },
    'fridge': {
        'windowlength': 599,
        'on_power_threshold': 50,
        'max_on_power': 3323,
        'mean': 200,
        'std': 400,
        's2s_length': 512,
        'houses': [1, 2, 3, 5, 7, 8, 9, 10, 12, 15, 17, 19, 20],
        'channels': [1, 1, 2, 1, 1, 1, 1, 4, 1, 1, 2, 1, 1],
        'train_house': [1, 2, 3, 5, 7, 8, 9, 10, 12],
        'test_house': [15],
        'validation_house': [17, 19, 20]
    },
    'dishwasher': {
        'windowlength': 599,
        'on_power_threshold': 10,
        'max_on_power': 3964,
        'mean': 700,
        'std': 1000,
        's2s_length': 1536,
        'houses': [1, 2, 3, 5, 6, 7, 9, 10, 13, 15, 16, 18, 20],
        'channels': [6, 3, 5, 4, 3, 6, 4, 6, 4, 4, 6, 6, 5],
        'train_house': [1, 2, 3, 5, 6, 7, 9, 10, 13],
        'test_house': [20],
        'validation_house': [15, 16, 18]
    },
    'washingmachine': {
        'windowlength': 599,
        'on_power_threshold': 20,
        'max_on_power': 3999,
        'mean': 400,
        'std': 700,
        's2s_length': 2000,
        'houses': [1, 2, 3, 5, 6, 7, 8, 9, 10, 13, 15, 16, 17, 18, 19, 20],
        'channels': [5, 2, 6, 3, 2, 5, 4, 3, 5, 3, 3, 5, 4, 5, 2, 4],
        'train_house': [1, 2, 3, 5, 6, 7, 9, 10, 13, 15, 16],
        'test_house': [8],
        'validation_house': [17, 18, 19, 20]
    }
}

appliance_name = 'microwave'
save_path = './new_data/' + appliance_name + '/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
windowlength = params_appliance[appliance_name]['windowlength']
offset = int(0.5*(windowlength-1.0))
batchsize = 100000
save_csv = pd.DataFrame(columns=list(range(windowlength))+['label'])
for batch_idx in range(0, 1*batchsize, batchsize):
    for building in params_appliance[appliance_name]['train_house']:
        print('Loading Building' + str(building))
        channel = params_appliance[appliance_name]['channels'][params_appliance[appliance_name]['houses'].index(building)]
        file_name = './data/refit/House_' + str(building) + '.csv'
        single_csv = pd.read_csv(file_name,
                                 nrows=batchsize,
                                 skiprows=int(batch_idx)*batchsize,
                                 header=0,
                                 names=['aggregate', 'label'],
                                 usecols=[2, channel+2],
                                 na_filter=False,
                                 parse_dates=True,
                                 infer_datetime_format=True,
                                 memory_map=True
                                 )
        single_csv['label'] = [int(item>params_appliance[appliance_name]['on_power_threshold']) for item in single_csv['label']]
        # Normalization
        single_csv['aggregate'] = (single_csv['aggregate'] - aggregate_mean) / aggregate_std
    
        num_of_ones = len(single_csv['label'][single_csv['label']>0])
        if (num_of_ones>0):
            n = len(single_csv)-2*offset
            print('Total length: '+str(n))
            save_csv_part = pd.DataFrame(columns=list(range(windowlength))+['label'])
            for idx in range(0,n):
                save_csv_part.loc[idx] = list(single_csv['aggregate'][idx:idx+windowlength])+[single_csv['label'][idx+offset]]
                if (idx % 1000 == 0):
                    print('Index: '+str(idx//1000))
            del single_csv
            print('Finish loading.')
            count_table = save_csv_part.groupby('label').count()
            print(count_table)
            print('Undersampling...')
            model_RandomUnderSampler=RandomUnderSampler()                #建立RandomUnderSample模型对象
            x_RandomUnderSample_resampled,y_RandomUnderSample_resampled=model_RandomUnderSampler.fit_sample(save_csv_part.iloc[:,:-1],save_csv_part.iloc[:,-1])         #输入数据并进行欠抽样处理
            del save_csv_part
            x_RandomUnderSample_resampled=pd.DataFrame(x_RandomUnderSample_resampled)
            y_RandomUnderSample_resampled=pd.DataFrame(y_RandomUnderSample_resampled,columns=['label'])
            RandomUnderSampler_resampled=pd.concat([x_RandomUnderSample_resampled,y_RandomUnderSample_resampled],axis=1)
            print(RandomUnderSampler_resampled.groupby('label').count())
            RandomUnderSampler_resampled.to_csv(save_path + 'train.csv', mode='a', index=False, header=False)
            del x_RandomUnderSample_resampled, y_RandomUnderSample_resampled, RandomUnderSampler_resampled
        else:
            print('Only 1 class so this dataset is dumped.')
            del single_csv
        # Save
        
    
    
# import numpy as np
# import matplotlib.pyplot as plt
# offset = int(0.5*198)
# savepred = np.load('G:/transferNILM-master/results/kettle/kettle_test_H2.csv_pred.npy')
# savegt = np.load('G:/transferNILM-master/results/kettle/kettle_test_H2.csv_gt.npy')
# savemains = np.load('G:/transferNILM-master/results/kettle/kettle_test_H2.csv_mains.npy')
# fig1 = plt.figure()
# random_num = np.random.randint(0,len(savegt)-1500)
# ax1 = fig1.add_subplot(111)
# ax1.plot(savemains[offset+random_num:offset+random_num+1500], color='#7f7f7f', linewidth=1.8)
# ax1.plot(savegt[random_num:random_num+1500], color='#d62728', linewidth=1.6)
# ax1.plot(savepred[random_num:random_num+1500],color='#1f77b4',linewidth=1.5)
# ax1.grid()
# ax1.set_title('Test results on kettle', fontsize=16, fontweight='bold', y=1.08)
# ax1.set_ylabel('W')
# ax1.legend(['aggregate', 'ground truth', 'prediction'])
# mng = plt.get_current_fig_manager()
# plt.show(fig1)

# import numpy as np
# import matplotlib.pyplot as plt
# offset = int(0.5*198)
# savepred = np.load('./results_new/kettle/pred.npy')
# savegt = np.load('./results_new/kettle/gt.npy')
# fig1 = plt.figure()
# random_num = np.random.randint(0,len(savegt)-1500)
# ax1 = fig1.add_subplot(111)
# ax1.plot(savegt[random_num:random_num+1500], color='#d62728', linewidth=2)
# ax1.plot(savepred[random_num:random_num+1500],color='#1f77b4',linewidth=1)
# ax1.grid()
# ax1.set_title('Test results on kettle', fontsize=16, fontweight='bold', y=1.08)
# ax1.set_ylabel('On_probability')
# ax1.legend(['ground truth', 'prediction'])
# mng = plt.get_current_fig_manager()
# plt.show(fig1)


# import nilmtk
# import matplotlib as plt
# import os
# DATA_PATH = 'G:/transferNILM-master/data/refit/'
# SAVE_PATH = 'G:/transferNILM-master/HDF5data.h5'

# nilmtk.dataset_converters.convert_refit(DATA_PATH, SAVE_PATH)
# REFIT = nilmtk.dataset.DataSet(SAVE_PATH)
# elec = REFIT.buildings[1].elec
# fraction = elec.submeters().fraction_per_meter().dropna()
# labels = elec.get_labels(fraction.index)
# plt.figure(figsize=(10,30))
# fraction.plot(kind='pie', labels=labels)
# elec.plot_when_on(on_power_threshold = 40)