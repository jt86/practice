__author__ = 'jt306'
import os, sys
import numpy as np
from Get_Full_Path import get_full_path
import logging

def get_madelon_data():
    print( 'getting madelon data')
    with open(get_full_path("Privileged_Data/MADELON/madelon_train.data"),"r+") as file:
        features_array = np.genfromtxt(file, dtype=None)
        features_array.shape=(2000,500)

    with open(get_full_path("Desktop/Privileged_Data/MADELON/madelon_train.labels"),"r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(2000)

    return features_array, labels_array


