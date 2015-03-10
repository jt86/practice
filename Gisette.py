__author__ = 'jt306'
import numpy as np

def get_gisette_data():
    print('Reading Gisette data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/GISETTE/gisette_train.data","r+") as file:
        features_array = np.genfromtxt(file, dtype=None)
        features_array.shape=(6000,5000)
        print "first row", features_array[0]

    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/GISETTE/gisette_train.labels","r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(6000)

    return features_array, labels_array
#########################
