__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import logging

def get_arcene_data(debug=False):
    if debug:

        logging.info('Reading Arcene SMALL data from disk')
        with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.data-small"), "r+") as infile:
            features_array = np.genfromtxt(infile, dtype=None)
            features_array.shape = (100, 100)


        with open(get_full_path("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/ARCENE/arcene_train.labels"), "r+") as infile:
            labels_array = np.genfromtxt(infile, dtype=None)
            labels_array.shape = (100)

        return features_array, labels_array

    logging.info('Reading Arcene data from disk')
    with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.data"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)
        features_array.shape = (100, 10000)


    with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.labels"), "r+") as infile:
        labels_array = np.genfromtxt(infile, dtype=None)
        labels_array.shape = (100)

    return features_array, labels_array

