__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import time

def get_arcene_data(debug=False):

    print('Reading Arcene data from disk')
    with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.data"), "r+") as infile:
        features_array = np.loadtxt(infile, dtype=float)
        features_array.shape = (100, 10000)
    with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.labels"), "r+") as infile:
        labels_array = np.loadtxt(infile, dtype=float)
        labels_array.shape = (100)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances



positive, negative = get_arcene_data()
print positive.shape
print negative.shape