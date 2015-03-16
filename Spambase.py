__author__ = 'jt306'

import csv
import numpy as np
from Get_Full_Path import get_full_path

def get_spambase_data(debug=False):
    print('Reading Spambase data from disk')
    with open(get_full_path("Privileged_Data/new_data/spambase.csv", "rU")) as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)

            # print row

        # print len(features_array), len(features_array[0])
        features_array = np.array(features_array)
        features_array.shape=(4601,58)

        features_array = np.array(features_array, dtype=float)

        # features_array = np.array([line for line in features_array if '?' not in line], dtype=np.int)

        labels_array = np.array(features_array[:,-1],dtype=int)
        labels_array[labels_array==0]=-1

        features_array = features_array[:,:-1]

    return features_array, labels_array

# get_spambase_data()