__author__ = 'jt306'

import csv
import numpy as np
from Get_Full_Path import get_full_path
import logging

def get_hillvalley_data(debug=False):
    logging.info('Reading HILLVALLEY data from disk')
    #with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/hillvalley.csv", "rU") as infile:
    with open(get_full_path("Desktop/Privileged_Data/new_data/hillvalley.csv"), "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        next(infile)        #to skip header row
        features_array +=(row for row in reader)
        features_array = np.array(features_array)
        features_array.shape=(606,101)
        labels_array = features_array[:,-1]

        labels_array[labels_array=='0']=-1

        labels_array=np.array(labels_array,dtype=float)


        features_array = np.add(np.array(features_array[:,:-1], dtype = float),1)

    return features_array, labels_array

