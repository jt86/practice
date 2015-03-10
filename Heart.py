__author__ = 'jt306'
import numpy as np


def get_heart_data(debug=False):
    print('Reading HEART data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/heart.dat", "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)
        # print features_array.size
        features_array=np.array(str(features_array).translate(None,'()').split(","),dtype=float)
        features_array.shape = (270, 14)

        # print "shape",features_array.shape
        labels_array = np.array(features_array[:,13])
        features_array = np.array(features_array[:,:13])
        # print labels_array, features_array[0]

        # print features_array.shape
        # print labels_array.shape


        labels_array[labels_array==2] = -1
        # print labels_array

        # print "\n\n\n"
        return features_array, labels_array


#
# data = open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/house-votes-84.data", "r+")
# data_array = np.loadtxt(data,delimiter=",")
# print data_array.shape
# labels_array = data_array[:,-1]
# labels_array[labels_array==0] = -1
