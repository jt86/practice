__author__ = 'jt306'
import csv
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from Imputer import imputer
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from Get_Full_Path import get_full_path

alpha_indices = [3,4,5,6,12]
int_indices = [index for index in range(15) if index not in alpha_indices]
# print alpha_indices
# print int_indices
# np.set_printoptions(threshold=np.nan)



def convert_to_ascii(text):
    return "".join(str(ord(char)) for char in text)

def get_crx_data(debug=False):
    print('Reading CRX data from disk')
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/crx.data", "rU") as infile:
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

        # print features_array.shape
        # print labels_array.shape

        labels_array[labels_array=='+']=1
        labels_array[labels_array=='-']=-1

        labels_array = np.array(labels_array, dtype=int)
        # print labels_array
        int_features = features_array[:,int_indices]


        alpha_features = features_array[:,alpha_indices]
        # print alpha_features[0]

        int_array= []
        for (i,value) in enumerate(alpha_features):
        # for line in alpha_features:
            # int_feats =
            int_array.append([convert_to_ascii(value) for value in alpha_features[i]])
        int_array = np.array(int_array, dtype=int)


        enc = OneHotEncoder()
        enc.fit(int_array)
        # print int_array[0]
        one_hot_array =  enc.transform(int_array).toarray()

        print "int feats shape",int_features.shape,"one hot shape", one_hot_array.shape

        features_array  = np.array(np.hstack((int_features, one_hot_array)),dtype=float)
        # print "Feats array shape",features_array.shape
        # print "first item in feats array",features_array[0]
        # print "Labels array shape", labels_array.shape
        # print "labels:", labels_array

        return features_array, labels_array

# get_crx_data()




        # int_features = features_array[:,int_indices]
        # non_int_features = features_array[:,non_int_indices]
        #
        # print int_features[0]
        # print non_int_features[0]

        #
        # list_of_dicts = []
        # for row in features_array:
        #     print "\n"
        #     row_dict = {}
        #
        #     i=0
        #     for feat in row:
        #
        #         row_dict[i]=feat
        #         i+=1
        #     print row_dict
        #     list_of_dicts.append(row_dict)

        # print list_of_dicts
        #
        # hasher = FeatureHasher(n_features=44)
        # X_new = hasher.fit_transform(list_of_dicts)