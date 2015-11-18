__author__ = 'jt306'
import numpy as np
from sklearn import preprocessing

def get_random_array(num_instances,num_feats):
    random_array = np.random.rand(num_instances,num_feats)
    random_array = preprocessing.scale(random_array)
    return random_array


get_random_array(2,3)