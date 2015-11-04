__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import csv
from sklearn.preprocessing import  Normalizer
from scipy import sparse as sp
from sklearn.cross_validation import StratifiedKFold
# import numpy.linalg.norm


def get_train_and_test_this_fold(dataset,datasetnum,k, skf_seed):	#N,test_N per class
    if dataset=='tech':
        class0_data,class1_data=get_techtc_data(datasetnum)
    class0_labels = [-1]*class0_data.shape[0]
    class1_labels = [1]* class1_data.shape[0]
    all_labels = np.r_[class0_labels, class1_labels]
    print (all_labels.shape)


    all_data = np.vstack([class0_data,class1_data])
    print(all_data.shape)

    skf = StratifiedKFold(all_labels, n_folds=4, random_state=skf_seed)
    for fold_num, (train_index, test_index) in enumerate(skf):
        if fold_num==k:
            train_data, test_data = all_data[train_index],all_data[test_index]
            train_labels, test_labels = all_labels[train_index], all_labels[test_index]


    #L1 normalization ============================
    normaliser = Normalizer(norm='l1')
    train_data=normaliser.fit_transform(train_data)
    normaliser = Normalizer(norm='l1')
    test_data=normaliser.fit_transform(test_data)

    print ('train data shape', train_data.shape, 'test data shape', test_data.shape)
    return np.asarray(train_data), np.asarray(test_data), np.asarray(train_labels), np.asarray(test_labels)




def get_techtc_data(dataset_index):
    max_num_of_feats,instances_count =0,0
    labels_list,all_instances = [],[]
    long_words_dict = get_longword_indices(dataset_index)
    max_num_of_feats= (len(long_words_dict))

    with open(get_tech_address(dataset_index), "r")as infile:
        for row_num, line in enumerate(infile):
            if row_num%2==1 and 1<row_num:
                row = line.split()
                label = row.pop(0)
                labels_list.append(int(label))
                array_of_tuples = list([item.split(':') for item in row])
                new_feats=[]
                for pair in array_of_tuples:                    #for each feature in a line
                    if int(pair[0]) in long_words_dict:         #if the feature isn't too short
                        new_tuple = [long_words_dict[int(pair[0])],pair[1]]      #make a new tuple of the index and the value
                        new_feats.append(new_tuple)             #add this to the new features
                instances_count+=1                              #add one for each training data point
                all_instances.append(new_feats)                 #add all the new features
        dok = sp.dok_matrix((instances_count, max_num_of_feats), dtype=int)
        for count, instance in enumerate(all_instances):
            for index, value in instance:
                dok[count,int(index)-1] = int(value)

    labels_list=np.array(labels_list)
    positive_indices = (labels_list==1).nonzero()
    negative_indices = (labels_list==-1).nonzero()
    dok=np.array(dok.todense(), dtype=float)
    positive_instances = dok[positive_indices]
    negative_instances = dok[negative_indices]
    print (positive_instances.shape, negative_instances.shape)

    return(positive_instances,negative_instances)



def get_tech_address(dataset_index):
    with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_49.txt"), "r") as infile:
        line = infile.readlines()[dataset_index]
        return (get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/{}/vectors.dat".format(line.strip('\r\n'))))


def get_words(dataset_index):
    with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_49.txt"), "r") as infile:
        line = infile.readlines()[dataset_index]
        return (get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/{}/features.idx".format(line.strip('\r\n'))))


#Iterates over list of words with indices; puts the original index of long words into a dict with its new index
def get_longword_indices(dataset_index):
    long_words = {}
    long_words_count = 0
    short_words = []
    with open(get_words(dataset_index), "r")as infile:
        for line_index, line in enumerate(infile):
            if line_index>8:
                index,word = line.split()[0],line.split()[1]
                if len(word)>4:
                    long_words[int(line_index-8)]=long_words_count
                    long_words_count+=1
                else:
                    short_words.append(word)
    return long_words

# get_train_and_test_this_fold('tech',0,1)