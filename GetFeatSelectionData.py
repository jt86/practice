'''
Get the data for one fold for Arcene, Gisette etc
'''


__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import csv
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, Normalizer
from scipy import sparse as sp
import sys
from sklearn.cross_validation import StratifiedKFold
# import numpy.linalg.norm
from sklearn import preprocessing


def get_train_and_test_this_fold(dataset,datasetnum,k, skf_seed):	#N,test_N per class
    if dataset=='arcene':
        class0_data, class1_data = get_arcene_data()
    if dataset=='madelon':
        class0_data, class1_data = get_madelon_data()
    if dataset=='gisette':
        class0_data, class1_data = get_gisette_data()
    if dataset=='dexter':
        class0_data, class1_data = get_dexter_data()
    if dataset=='dorothea':
        class0_data, class1_data = get_dorothea_data()

    class0_labels = [-1]*class0_data.shape[0]
    class1_labels = [1]* class1_data.shape[0]
    all_labels = np.r_[class0_labels, class1_labels]
    print (all_labels.shape)


    all_data = np.vstack([class0_data,class1_data])
    print('all data (Both classes) shape:',all_data.shape)


    skf = StratifiedKFold(all_labels, n_folds=10, random_state=skf_seed)
    for fold_num, (train_index, test_index) in enumerate(skf):
        if fold_num==k:
            train_data, test_data = all_data[train_index],all_data[test_index]
            train_labels, test_labels = all_labels[train_index], all_labels[test_index]


    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)


    # train_data = train_data/np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, train_data).reshape(-1,1)
    # test_data = test_data/np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, test_data).reshape(-1,1)
    print ('train data shape', train_data.shape, 'test data shape', test_data.shape)
    return np.asarray(train_data), np.asarray(test_data), np.asarray(train_labels), np.asarray(test_labels)



def get_arcene_data():
    print('Reading Arcene data from disk')
    with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.data"), "r+") as infile:
        features_array = np.loadtxt(infile, dtype=float)
        features_array.shape = (100, 10000)
    with open(get_full_path("Desktop/Privileged_Data/ARCENE/arcene_train.labels"), "r+") as infile:
        labels_array = np.loadtxt(infile, dtype=float)
        labels_array.shape = (100)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances

# arcene_data1,arcene_data2 = get_arcene_data()

# train_data, test_data, train_labels, test_labels = get_train_and_test_this_fold('arcene',14,30)
# print train_data.shape, test_data.shape, train_labels.shape, test_labels.shape

def get_madelon_data():
    print( 'getting madelon data')
    with open(get_full_path("Desktop/Privileged_Data/MADELON/madelon_train.data"),"r+") as file:
        features_array = np.loadtxt(file, dtype=float)
        features_array.shape=(2000,500)

    with open(get_full_path("Desktop/Privileged_Data/MADELON/madelon_train.labels"),"r+") as file:
        labels_array = np.loadtxt(file, dtype=float)
        labels_array.shape=(2000)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances


def get_gisette_data():
    print('Reading Gisette data from disk')
    with open(get_full_path("Desktop/Privileged_Data/GISETTE/gisette_train.data"), "r+") as file:
        features_array = np.loadtxt(file, dtype=float)
        features_array.shape = (6000, 5000)

    with open(get_full_path("Desktop/Privileged_Data/GISETTE/gisette_train.labels"), "r+") as file:
        labels_array = np.loadtxt(file, dtype=float)
        labels_array.shape = (6000)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances



def get_dexter_data():
    print( "Getting DEXTER data")
    features_array = np.zeros((300,20000))
    fh = open(get_full_path("Desktop/Privileged_Data/DEXTER/dexter_train.data"),"rU")
    for row_num, line in enumerate(fh):
        row = line.split(' ')      #make list of numbers for each instance
        for item in row:
            if len(item)>1:
                index, value = item.split(':')[0],item.split(':')[1]
                features_array[row_num,index]=value
    with open(get_full_path("Desktop/Privileged_Data/DEXTER/dexter_train.labels"),"r+") as file:
        labels_array=np.empty(shape=[300,1])
        for row_num, line in enumerate(file):
            labels_array[row_num] = line
        labels_array.shape=(300)
    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances




def get_dorothea_data():
    print( "Getting DOROTHEA data")
    with open(get_full_path("Desktop/Privileged_Data/DOROTHEA/dorothea_train.data"),"rU") as fh:
        features_array = np.zeros([800, 100000])
        for row_num, line in enumerate(fh):
            row = line.split('\t')      #make list of numbers for each instance
            # print('row',row)
            indices_of_1s = np.array([int(r)-1 for r in row if r.isdigit()])
            features_array[row_num,indices_of_1s]=1
    with open(get_full_path("Desktop/Privileged_Data/DOROTHEA/dorothea_train.labels"),"r+") as file:
        labels_array=np.empty(shape=[800,1])
        print (labels_array.shape)
        # print (labels_array)
        for row_num, line in enumerate(file):
            labels_array[row_num] = line
        print (labels_array.shape)

    labels_array.shape=(800)
    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return (positive_instances, negative_instances)


