__author__ = 'jt306'
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder


def get_cancer_data(debug=False):
    print('Reading Cancer data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/cancer.csv", "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)

            # print row
        #
        print len(features_array), len(features_array[0])
        features_array = np.array(features_array)
        features_array.shape=(32,57)

        features_array = np.array(features_array)

        features_array = np.array([line for line in features_array if '?' not in line], dtype=np.int)

        labels_array = features_array[:,0]
        labels_array=one_v_rest(labels_array,2)
        # print labels_array


        features_array = features_array[:,1:]
        # print labels_array
        # print labels_array.shape
        # print features_array.shape




        enc = OneHotEncoder()
        enc.fit(features_array)
        features_array =  enc.transform(features_array).toarray()



        print "Feats array shape",features_array.shape
        print "first item in feats array",features_array[0]
        print "Labels array shape", labels_array.shape
        print "labels:", labels_array

        return features_array, labels_array

def one_v_rest(labels_array, chosen_one):
    for index, item in enumerate(labels_array):
        if item == chosen_one:
            print 'match!'
            labels_array[index] = 1
        else:
            labels_array[index]= (-1)
    return labels_array

# get_cancer_data()