from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from Get_Full_Path import get_full_path
import numpy as np
import csv
from pandas import get_dummies

numerical_value_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15,
        'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26 ,'?':27}


def get_mushroom_data():

    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/mushroom.data.csv", "r+") as infile:
        features_array = []
        reader = csv.reader(infile, dialect=csv.excel_tab)
        for row in reader:
            features_array.append(str(row).translate(None, "[]'").split(","))

    features_array = np.array(features_array)
    labels_array = np.array(features_array[:, 0], dtype=int)
    features_array=np.array(features_array[:,1:])
    labels_array = np.reshape(labels_array, 8124)

    labels_array[labels_array=='e']=1
    labels_array[labels_array=='p']=-1


    for item in numerical_value_dict:
        features_array[features_array==item]=numerical_value_dict[item]
        print item

    print features_array[0]
    # dummies = get_dummies(features_array)

    print features_array.shape, labels_array.shape
    print labels_array

    enc = OneHotEncoder(n_values='auto', categorical_features ='all')
    features_array = enc.fit_transform(features_array)
    print features_array[0]
    print 'feats array shape',features_array.shape

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    print positive_instances.shape, negative_instances.shape
    return positive_instances, negative_instances

get_mushroom_data()