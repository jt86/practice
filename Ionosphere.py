__author__ = 'jt306'

import csv
import numpy as np
from Get_Full_Path import get_full_path

def get_ionosphere_data(debug=False):
    print('Reading IONOSPHERE data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/ionosphere.csv"), "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)


            # print row

        features_array = np.array(features_array)
        features_array.shape=(351,35)

        # features_array = np.array([line for line in features_array if '?' not in line], dtype=np.int)

        labels_array = features_array[:,-1]
        labels_array[labels_array=='g']=1
        labels_array[labels_array=='b']=-1
        labels_array=np.array(labels_array,dtype=float)

        features_array = np.add(np.array(features_array[:,:-1], dtype = float),1)

    return features_array, labels_array

# get_ionosphere_data()

