__author__ = 'jt306'
import numpy as np
from scipy import sparse as sp
import logging

def get_dexter_data():
    logging.info( "Getting DEXTER data")
    dok = sp.dok_matrix((300, 20000), dtype=int)

    fh = open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/DEXTER/dexter_train.data","rU")

    line = fh.next().strip()
    for row_num, line in enumerate(fh):

        row = line.split('\t')      #make list of numbers for each instance
        logging.info( row)
        indices_of_1s = np.array([int(r)-1 for r in row if r.isdigit()])           #make integers and put in array

        dok[row_num,indices_of_1s] = 1



    features_array = dok.todense()    #csr format




    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/DEXTER/dexter_train.labels","r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(300)

    return features_array, labels_array