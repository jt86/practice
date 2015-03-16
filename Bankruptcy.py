__author__ = 'jt306'
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Get_Full_Path import get_full_path

def convert_to_ascii(text):
    return "".join(str(ord(char)) for char in text)

def get_bankruptcy_data(debug=False):
    print('Reading Bankruptcy data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/Qualitative_Bankruptcy.data.csv"), "rU") as infile:
        features_array = []
        reader = csv.reader(infile,delimiter = ',')
        features_array +=(row for row in reader)

            # print row

        # print len(features_array), len(features_array[0])
        features_array = np.array(features_array)
        features_array.shape=(250,7)

        features_array = np.array(features_array)

        features_array = np.array([line for line in features_array if '?' not in line])# dtype=np.int

        labels_array = features_array[:,-1]

        labels_array[labels_array=='NB']=-1
        labels_array[labels_array=='B']=1
        labels_array = np.array(labels_array, dtype=int)
        # print labels_array


        features_array = features_array[:,:-1]


        int_array= []
        for (i,value) in enumerate(features_array):
            int_array.append([convert_to_ascii(value) for value in features_array[i]])
        features_array = np.array(int_array, dtype=int)


        enc = OneHotEncoder()
        enc.fit(features_array)
        features_array =  enc.transform(features_array).toarray()


        return features_array, labels_array


