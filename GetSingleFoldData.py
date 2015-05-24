__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path


def get_train_and_test_this_fold(dataset):	#N,test_N per class
    
    if dataset=='arcene':
        class0_data, class1_data = get_arcene_data()
        N, test_N =14,30
    if dataset=='madelon':
        class0_data, class1_data = get_madelon_data()
        N, test_N = 250,250
    if dataset=='gisette':
        class0_data, class1_data = get_gisette_data()
        N, test_N = 100,200
    if dataset=='dexter':
        class0_data, class1_data = get_dexter_data()
        N, test_N = 50,100
    if dataset=='dorothea':
        class0_data, class1_data = get_dorothea_data()
        N, test_N = 25,50

    if (N+test_N > class0_data.shape[0]) or (N+test_N > class1_data.shape[0]):
        print "Warning: total number of samples is less than required ", class0_data.shape[0], class1_data.shape[0]
        N=44; test_N = 44

    idx1 = np.random.permutation(class0_data.shape[0])
    train1, test1 = idx1[:N], idx1[N:N+test_N]
    idx2 = np.random.permutation(class1_data.shape[0])
    train2, test2 = idx2[:N], idx2[N:N+test_N]

    train_data = np.r_[class0_data[train1], class1_data[train2]]
    test_data = np.r_[class0_data[test1], class1_data[test2]]
    train_labels = np.ravel(np.r_[[1]*N, [-1]*N])
    test_labels = np.ravel(np.r_[[1]*test_N, [-1]*test_N])

    #L1 normalization ============================
    train_data = train_data/np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, train_data).reshape(-1,1)
    test_data = test_data/np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, test_data).reshape(-1,1)

    return np.asarray(train_data), np.asarray(test_data), np.asarray(train_labels), np.asarray(test_labels)


def get_arcene_data(debug=False):
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

#
# def get_dexter_data():
#     __author__ = 'jt306'
# import numpy as np
# from scipy import sparse as sp
# import logging
# from Get_Full_Path import get_full_path

def get_dexter_data():
    print( "Getting DEXTER data")
    dok = sp.dok_matrix((300, 20000), dtype=int)
    fh = open(get_full_path("Desktop/Privileged_Data/DEXTER/dexter_train.data"),"rU")
    line = fh.next().strip()
    for row_num, line in enumerate(fh):
        row = line.split('\t')      #make list of numbers for each instance
        indices_of_1s = np.array([int(r)-1 for r in row if r.isdigit()])           #make integers and put in array
        dok[row_num,indices_of_1s] = 1
    features_array = dok.todense()    #csr format

    with open(get_full_path("Desktop/Privileged_Data/DEXTER/dexter_train.labels"),"r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(300)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances

def get_dorothea_data():
    print( "Getting DOROTHEA data")
    dok = sp.dok_matrix((800, 100000), dtype=int)
    fh = open(get_full_path("Desktop/Privileged_Data/DOROTHEA/dorothea_train.data"),"rU")
    line = fh.next().strip()
    for row_num, line in enumerate(fh):
        row = line.split('\t')      #make list of numbers for each instance
        indices_of_1s = np.array([int(r)-1 for r in row if r.isdigit()])           #make integers and put in array
        dok[row_num,indices_of_1s] = 1
    features_array = dok.todense()    #csr format

    with open(get_full_path("Desktop/Privileged_Data/DOROTHEA/dorothea_train.labels"),"r+") as file:
        labels_array = np.genfromtxt(file, dtype=None)
        labels_array.shape=(800)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances
#
# positive_instances, negative_instances = get_madelon_data()
# print positive_instances.shape, negative_instances.shape