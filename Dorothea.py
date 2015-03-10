__author__ = 'jt306'
from scipy import sparse as sp
import numpy as np

# dok = sp.dok_matrix((800, 139351), dtype=bool)


def get_dorothea_data():
    print "Getting DOROTHEA data"
    dok = sp.dok_matrix((800, 100000), dtype=int)

    fh = open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/DOROTHEA/dorothea_train.data","rU")

    line = fh.next().strip()
    for row_num, line in enumerate(fh):

        row = line.split('\t')      #make list of numbers for each instance
        indices_of_1s = np.array([int(r)-1 for r in row if r.isdigit()])           #make integers and put in array

        dok[row_num,indices_of_1s] = 1



    features_array = dok.todense()    #csr format




    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/DOROTHEA/dorothea_train.labels","r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(800)

    return features_array, labels_array