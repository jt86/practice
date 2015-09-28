__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import logging

def get_vote_data(debug=False):
    print('Reading VOTE data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/house-votes-84.data"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)

        features_array.shape = (435, 17)

        d = {'republican':1.,'democrat':-1., 'y':1., 'n':0., '?':0.5}
        new_array = np.copy(features_array)
        for k, v in d.items():
            new_array[features_array==k] = v


        labels_array = np.array(new_array[:,0], dtype=float)
        features_array = np.array(new_array[:,1:], dtype=float)

        return features_array, labels_array


