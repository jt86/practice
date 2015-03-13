__author__ = 'jt306'

import csv
import numpy as np
from Get_Full_Path import get_full_path

def get_musk1_data(debug=False):
    print('Reading Musk1 data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/musk1.csv", "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)


            # print row

        print len(features_array), len(features_array[0])
        features_array = np.array(features_array)
        features_array.shape=(476,169)

        # features_array = np.array([line for line in features_array if '?' not in line], dtype=np.int)

        labels_array = np.array(features_array[:,-1],dtype=float)
        labels_array[labels_array==0]=-1
        print type(labels_array[0])
        features_array = np.add(np.array(features_array[:,2:-1], dtype = float),472)
        print features_array.shape
        print labels_array.shape
        print 'labels array',labels_array
        print 'num of labels',labels_array.shape
        print 'feat array shape',features_array.shape
        # print 'first instance',features_array[0]
    return features_array, labels_array

# get_musk2_data()

