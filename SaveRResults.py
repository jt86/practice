import os
import numpy as np
path = '/Volumes/LocalDataHD/j/jt/jt306/Desktop/results/GPC_conf'
count=0
from Get_Full_Path import get_full_path
from matplotlib import pyplot as plt

def save_r_results():
    all_results = []
    for num_selected in [300,500]:
        for datasetnum in range(49):
            print('\n')
            all_folds_single_setting =[]
            for seed in range(10):
                for fold in range(10):
                    location = 'error-100-{}-tech-{}-{}-{}-.txt'.format(num_selected, datasetnum, seed, fold)
                    full_path = os.path.join(path,location)
                    if os.path.exists(full_path):
                        with open(full_path) as savefile:
                            # print(num_selected,datasetnum,seed,fold)
                            value = (savefile.readline())
                            # print('value',value)
                            if value=='':
                                print ('print ("{} {} {} {}")'.format(num_selected,datasetnum,seed,fold))
    #                         score = float(value)
    #                         all_folds_single_setting.append(score)
    #                         print (len(all_folds_single_setting))
    #         all_results+=([all_folds_single_setting])
    #         # print ('len all results',len(all_results))
    #
    # all_results = np.array(all_results)
    # print (all_results.shape)
    # means = np.mean(all_results,axis=1)
    # print (means)
    #
    # print(all_results.shape)
    # np.save(os.path.join(path,'mean_results'),means)

save_r_results()

# gpc_conf_results = np.load(os.path.join(path,'mean_results.npy'))
# gpc_conf_300 = np.array([1- item for item in gpc_conf_results[:40]])
# print (gpc_conf_300.shape)
#
# experiment_name = '10x10-ALLCV-3to3-featsscaled-300'
# list_of_baselines= np.mean(np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-baseline.npy'.format(experiment_name)))[9:],axis=1)
# list_of_300_rfe = np.mean(np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-rfe.npy'.format(experiment_name)))[9:],axis=1)
# list_of_300_lupi = np.mean(np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-lupi.npy'.format(experiment_name)))[9:],axis=1)
#
# print (len(list_of_300_rfe))
#
# gpc_conf_300=gpc_conf_300[np.argsort(list_of_baselines)]
# list_of_300_rfe=list_of_300_rfe[np.argsort(list_of_baselines)]
# list_of_300_lupi=list_of_300_lupi[np.argsort(list_of_baselines)]
# list_of_baselines=list_of_baselines[np.argsort(list_of_baselines)]
#
# plt.plot(range(40),list_of_300_lupi,color='red',label='LUFe SVM+')
# plt.plot(range(40),list_of_baselines,color='black',label='all features SVM baseline')
# plt.plot(range(40),list_of_300_rfe,color='blue',label='top 300 only SVM baseline')
# plt.plot(range(40),gpc_conf_300,color='green',label='LUFe GPC_conf (new setting)')
# plt.legend(loc='best')
# plt.ylabel('accuracy')
# plt.xlabel('dataset index')
# plt.savefig('/Volumes/LocalDataHD/j/jt/jt306/Desktop/prelim_results')
# plt.show()
