__author__ = 'jt306'

from Get_Full_Path import get_full_path
import numpy as np
from GetTechData import get_techtc_data
import matplotlib.pyplot as plt
num_datasets=49


#load results for baseline, rfe, lupi, as 10x10 arrays, saved using SaveResults.py
experiment_name = '10x10-ALLCV-3to3-featsscaled-300'
list_of_baselines= np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-baseline.npy'.format(experiment_name)))
list_of_300_rfe = np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-rfe.npy'.format(experiment_name)))
list_of_300_lupi = np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-lupi.npy'.format(experiment_name)))


list1 = list_of_baselines
list2 = list_of_300_lupi

setting1 =np.array([1-mean for mean in np.mean(list1,axis=1)])
setting2 = np.array([1-mean for mean in np.mean(list2,axis=1)])
# setting2 = setting2[np.argsort(setting1)]
# setting1 = setting1[np.argsort(setting1)]

####################################### Scatter plot: feats vs instances

rfe_better,lupi_better = [], []
for index,(setting1_error, setting2_error) in enumerate(zip(setting1,setting2)):
    if setting1_error < setting2_error:
        rfe_better.append(index)
    if setting1_error > setting2_error:
        lupi_better.append(index)
np.save(get_full_path('Desktop/Privileged_Data/all-results/rfe_better-{}'.format(experiment_name)),rfe_better)
np.save(get_full_path('Desktop/Privileged_Data/all-results/lupi_better-{}'.format(experiment_name)),lupi_better)

print('rfe better',len(rfe_better),'lupi better',len(lupi_better))

rfe_better_instances, rfe_better_features = [],[]
for dataset_index in rfe_better:
    features_array,labels_array=get_techtc_data(dataset_index)
    print (dataset_index,features_array.shape,labels_array.shape)
    rfe_better_instances.append(features_array.shape[0])
    rfe_better_features.append(features_array.shape[1])


lupi_better_features, lupi_better_instances = [],[]
for dataset_index in lupi_better:
    features_array,labels_array=get_techtc_data(dataset_index)
    print (dataset_index,features_array.shape,labels_array.shape)
    lupi_better_instances.append(features_array.shape[0])
    lupi_better_features.append(features_array.shape[1])

plt.scatter(rfe_better_features,rfe_better_instances,color='blue',label='LUPI not helpful')
plt.scatter(lupi_better_features,lupi_better_instances,color='red',label='LUPI improvement')
plt.xlabel('Number of features')
plt.ylabel('Number of instances')
plt.legend(loc='best')
plt.show()

####################################### Scatter plot:  vs

improvements_list = []
for setting1_error, setting2_error in zip(setting1,setting2):
    improvements_list.append(setting1_error-setting2_error)
