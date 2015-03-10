__author__ = 'jt306'
import numpy as np


def get_haberman_data(debug=False):
    print('Reading HABERMAN data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/haberman.data", "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)
        print features_array.shape
        print features_array[0]
        features_array.shape = (306, 4)
        print features_array.shape


        labels_array = features_array[:,3]
        features_array = features_array[:,:3]
        labels_array[labels_array==2]=-1
        print "feats shape", features_array.shape
        print "labels shape", labels_array.shape


        print "\n\n\n"
        return features_array, labels_array

