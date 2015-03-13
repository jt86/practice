__author__ = 'jt306'

import csv
import numpy as np
from Get_Full_Path import get_full_path

def get_wine_data(debug=False):
    print('Reading WINE data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/wine.csv", "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)
        features_array = np.array(features_array)

        features_array.shape=(178,14)
        labels_array = features_array[:,0]

        labels_array[labels_array=='3']=1
        labels_array[labels_array=='2']=-1

        labels_array=np.array(labels_array,dtype=float)

        features_array = np.array(features_array[:,1:], dtype = float)
        # features_array = np.add(np.array(features_array[:,:-1], dtype = float),1)
        # print features_array.shape
        # print labels_array.shape
        # print 'labels array',labels_array

        # print 'feat array shape',features_array.shape
        # print 'first instance',features_array[0]
    return features_array, labels_array

# get_wine_data()

