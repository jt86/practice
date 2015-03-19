__author__ = 'jt306'
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Imputer import imputer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from Get_Full_Path import get_full_path
import logging
alpha_indices = [3,4,5,6,12]
int_indices = [index for index in range(15) if index not in alpha_indices]
# logging.info( alpha_indices)
# logging.info( int_indices)
# np.set_logging.info(options(threshold=np.nan))



def convert_to_ascii(text):
    return "".join(str(ord(char)) for char in text)

def get_crx_data(debug=False):
    logging.info('Reading CRX data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/crx.data"), "rU") as infile:
        features_array = []
        reader = csv.reader(infile,dialect=csv.excel_tab)
        for row in reader:
            features_array.append(str(row).translate(None,"[]'").split(", "))
        features_array = np.array(features_array)


        #FOLLOWING LINE TO GET RID OF ALL WITH UNKNOWN VALUES
        features_array = np.array([line for line in features_array if '?' not in line])
        # features_array=imputer(features_array)

        labels_array = features_array[:,15]

        features_array = features_array[:,:15]

        labels_array[labels_array=='+']=1
        labels_array[labels_array=='-']=-1

        labels_array = np.array(labels_array, dtype=int)
        int_features = features_array[:,int_indices]


        alpha_features = features_array[:,alpha_indices]


        int_array= []
        for (i,value) in enumerate(alpha_features):
            int_array.append([convert_to_ascii(value) for value in alpha_features[i]])
        int_array = np.array(int_array, dtype=int)


        enc = OneHotEncoder()
        enc.fit(int_array)
        # logging.info( int_array[0])
        one_hot_array =  enc.transform(int_array).toarray()


        features_array  = np.array(np.hstack((int_features, one_hot_array)),dtype=float)


        return features_array, labels_array
