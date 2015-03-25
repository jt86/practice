__author__ = 'jt306'
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Get_Full_Path import get_full_path
import logging

def get_cancer_data(debug=False):
    print('Reading Cancer data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/cancer.csv"), "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)

        features_array = np.array(features_array)
        features_array.shape=(32,57)

        features_array = np.array(features_array)

        features_array = np.array([line for line in features_array if '?' not in line], dtype=np.int)

        labels_array = features_array[:,0]
        labels_array=one_v_rest(labels_array,2)

        features_array = features_array[:,1:]

        enc = OneHotEncoder()
        enc.fit(features_array)
        features_array =  enc.transform(features_array).toarray()

        return features_array, labels_array

def one_v_rest(labels_array, chosen_one):
    for index, item in enumerate(labels_array):
        if item == chosen_one:

            labels_array[index] = 1
        else:
            labels_array[index]= (-1)
    return labels_array

