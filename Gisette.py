__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import logging


def get_gisette_data():
    print('Reading Gisette data from disk')
    with open(get_full_path("Desktop/Privileged_Data/GISETTE/gisette_train.data"), "r+") as file:
        features_array = np.genfromtxt(file, dtype=None)
        features_array.shape = (6000, 5000)

    with open(get_full_path("Desktop/Privileged_Data/GISETTE/gisette_train.labels"), "r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape = (6000)

    return features_array, labels_array

# ########################
