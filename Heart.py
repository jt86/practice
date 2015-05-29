__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import logging


def get_heart_data(debug=False):
    print('Reading HEART data from disk')

    with open(get_full_path("Desktop/Privileged_Data/new_data/heart.dat"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)
        features_array = np.array(str(features_array).translate(None, '()').split(","), dtype=float)
        features_array.shape = (270, 14)
        labels_array = np.array(features_array[:, 13])
        features_array = np.array(features_array[:, :13])
        labels_array[labels_array == 2] = -1

        return features_array, labels_array

features_array,labels_array=get_heart_data()
print features_array.shape, labels_array.shape
#
# data = open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/house-votes-84.data", "r+")
# data_array = np.loadtxt(data,delimiter=",")
# logging.info( data_array.shape)
# labels_array = data_array[:,-1]
# labels_array[labels_array==0] = -1
