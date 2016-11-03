__author__ = 'jt306'

from Get_Full_Path import get_full_path
import numpy as np
from GetSingleFoldData import get_techtc_data
import matplotlib.pyplot as plt
num_datasets=49
import os
try:
    user_paths = os.environ['PYTHONPATH'].split(os.pathsep)
except KeyError:
    user_paths = []
print(user_paths)

#load results for baseline, rfe, lupi, as 10x10 arrays, saved using SaveResults.py
experiment_name = '10x10-ALLCV-3to3-featsscaled-300'
list_of_baselines= np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-baseline.npy'.format(experiment_name)))
list_of_300_rfe = np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-rfe.npy'.format(experiment_name)))
list_of_300_lupi = np.load(get_full_path('Desktop/Privileged_Data/all-results/{}-lupi.npy'.format(experiment_name)))


#### CHANGE THIS to change what is being compared
setting1 =np.array([1-mean for mean in np.mean(list_of_300_rfe,axis=1)])
setting2 = np.array([1-mean for mean in np.mean(list_of_300_lupi,axis=1)])


####################################### Scatter plot:  vs

#get improvements
improvements_list = []
for setting1_error, setting2_error in zip(setting1,setting2):
    improvements_list.append(setting1_error-setting2_error)

#get numbers of features and instances, and ratios
features, instances, ratios = [],[],[]
for dataset_index in range(49):
    print (dataset_index)
    features_array,labels_array=get_techtc_data(dataset_index)
    print (dataset_index,features_array.shape,labels_array.shape)
    instances.append(features_array.shape[0])
    features.append(features_array.shape[1])
    ratios.append(features_array.shape[1]/features_array.shape[0])
    print(dataset_index,instances,features,ratios)


#plot ratios
# plt.scatter(ratios,improvements_list,color='blue')
# plt.plot(ratios, np.poly1d(np.polyfit(ratios, improvements_list, 1))(ratios))
# plt.xlabel('Ratio of features:instances')
# plt.ylabel('LUPI improvement over RFE')
# plt.show()

#plot features
# plt.scatter(features,improvements_list,color='blue')
# plt.plot(features, np.poly1d(np.polyfit(features, improvements_list, 1))(features))
# plt.xlabel('Number of features')
# plt.ylabel('LUPI improvement over RFE')
# plt.show()

#plot instances
plt.scatter(instances,improvements_list,color='blue')
plt.plot(instances, np.poly1d(np.polyfit(instances, improvements_list, 1))(instances))
plt.xlabel('Number of instances')
plt.ylabel('LUPI improvement over RFE')
plt.show()
