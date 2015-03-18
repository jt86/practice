__author__ = 'jt306'
import numpy as np
from Get_Full_Path import  get_full_path

def get_haberman_data(debug=False):
    print('Reading HABERMAN data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/haberman.data"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)

        features_array.shape = (306, 4)



        labels_array = features_array[:,3]
        features_array = features_array[:,:3]
        labels_array[labels_array==2]=-1

        return features_array, labels_array

