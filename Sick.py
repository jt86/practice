__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import csv
from sklearn.preprocessing import OneHotEncoder, LabelEncoder



def get_sick_data():
    print('Reading Sick data from disk')
    with open(get_full_path("Desktop/Privileged_Data/Sick/sick.data.txt"), "r+") as infile:
        features_array = np.loadtxt(infile, dtype=str)
        features_array.shape = (3772, 30)
    labels_array=np.zeros(3772)
    for index,item in enumerate(features_array[:,-1]):
        if 'sick' in item:
            labels_array[index]=1
        else:
            labels_array[index]=-1

    with open(get_full_path("Desktop/Privileged_Data/Sick/sick2.txt"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=float,delimiter="\t", filling_values=np.NaN)
        print features_array.shape
        print features_array[0]
        features_array.shape = (3772, 29)

    final_feat=features_array[:,-1]
    final_feat.shape=(3772,1)
    features_array = features_array[:,:-1]

    one_hot_array = np.zeros((3772,5))
    for index,feature in enumerate(final_feat):
        one_hot_array[index,int(feature)-1]=1


    features_array=np.hstack((features_array,one_hot_array))
    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])

    return positive_instances, negative_instances

print get_sick_data()[0].shape, get_sick_data()[1].shape