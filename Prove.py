__author__ = 'jt306'
from Get_Full_Path import  get_full_path
import numpy as np
def get_prove_data():
    print('Reading Arcene data from disk')
    with open(get_full_path("Desktop/Privileged_Data/ml-prove/train.csv"), "r+") as infile:
        features_array = np.loadtxt(infile,dtype=str,delimiter=',')#, dtype=float)
    print len(features_array[0])
    print len(features_array)

    features_array=features_array[:,-6]
    print features_array.shape

    #
    # positive_instances = (features_array[labels_array==1])
    # negative_instances = (features_array[labels_array==-1])
    # return positive_instances, negative_instances

get_prove_data()