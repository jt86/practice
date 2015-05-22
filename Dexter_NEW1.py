__author__ = 'jt306'
import numpy as np
from scipy import sparse as sp
import logging
from Get_Full_Path import get_full_path

def get_dexter_data():
    print( "Getting DEXTER data")
    dok = sp.dok_matrix((300, 20000), dtype=int)

    fh = open(get_full_path("Desktop/Privileged_Data/DEXTER/dexter_train.data"),"rU")

    line = fh.next().strip()
    for row_num, line in enumerate(fh):

        row = line.split('\t')      #make list of numbers for each instance
        logging.info( row)
        indices_of_1s = np.array([int(r)-1 for r in row if r.isdigit()])           #make integers and put in array

        dok[row_num,indices_of_1s] = 1

    features_array = dok.todense()    #csr format




    with open(get_full_path("Desktop/Privileged_Data/DEXTER/dexter_train.labels"),"r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(300)

    return features_array, labels_array

print get_dexter_data()[0].shape, get_dexter_data()[1].shape