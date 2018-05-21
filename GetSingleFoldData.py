'''
Called by SingleFoldSlice.py and SingleFoldUnivariate.py
Gets the subset of data for one fold, for one setting
'''


__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
from scipy import sparse as sp
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing
# from GetFeatSelectionData import get_arcene_data,get_madelon_data,get_gisette_data,get_dexter_data,get_dorothea_data
# import numpy.linalg.norm
import sys
np.set_printoptions(linewidth=132)
import sklearn
import os.path
from os import mkdir
from GetFeatSelectionData import get_arcene_data, get_madelon_data, get_gisette_data, get_dexter_data, get_dorothea_data
from ReadInBBC import get_bbc_data
print('The scikit-learn version is {}.'.format(sklearn.__version__))
#
def load_dataset_from_name(dataset,datasetnum):
    if dataset=='tech':
         class0_data = np.load(get_full_path("Desktop/Privileged_Data/SavedDatasets/{}/{}positive.npy".format(dataset,datasetnum)))
         class1_data = np.load(get_full_path("Desktop/Privileged_Data/SavedDatasets/{}/{}negative.npy".format(dataset,datasetnum)))

    if dataset=='bbc':
        class0_data, class1_data = get_bbc_data()

    if dataset == 'arcene':
        class0_data, class1_data = get_arcene_data()
    if dataset == 'madelon':
        class0_data, class1_data = get_madelon_data()
    if dataset == 'gisette':
        class0_data, class1_data = get_gisette_data()
    if dataset == 'dexter':
        class0_data, class1_data = get_dexter_data()
    if dataset == 'dorothea':
        class0_data, class1_data = get_dorothea_data()
    return class0_data,class1_data





def get_train_and_test_this_fold(s):	#N,test_N per class

    class0_data, class1_data = load_dataset_from_name(s.dataset, s.datasetnum)
    class0_labels = [-1]*class0_data.shape[0]
    class1_labels = [1]* class1_data.shape[0]
    all_labels = np.r_[class0_labels, class1_labels]
    all_data = np.vstack([class0_data,class1_data])

    folder = get_full_path('Desktop/Privileged_Data/SavedTrainTestIndices/{}{}'.format(s.dataset, s.datasetnum))
    # if s.percentageofinstances!=100:
    #     folder = get_full_path('Desktop/Privileged_Data/SavedTrainTestIndices/{}{}-{}pc'.format(s.dataset, s.datasetnum, s.percentageofinstances))
    train_index = np.load(os.path.join(folder,'{}-{}-train.npy'.format(s.skfseed,s.foldnum)))
    test_index = np.load(os.path.join(folder,'{}-{}-test.npy'.format(s.skfseed, s.foldnum)))

    # uses train index and test index (loaded in previous lines) and uses these to slice from all_data
    train_data, test_data = all_data[train_index], all_data[test_index]
    train_labels, test_labels = all_labels[train_index], all_labels[test_index]

    print('train shape before',train_data.shape,'test shape',test_data.shape)

    train_data = preprocessing.scale(train_data)
    test_data = preprocessing.scale(test_data)
    print(train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)

    # num_to_take = len(class0_labels)*s.percentageofinstances//100
    # indices  = np.random.randint(len(class0_labels), size=num_to_take)
    # train_data, train_labels = train_data[indices], train_labels[indices]
    # print('\n shapes after:', train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    # sys.exit()

    # if using <100% training data, use a single fold of SKF to take balanced train/test subset
    if s.percentageofinstances != 100:
        skf1 = StratifiedKFold(train_labels, n_folds=10, shuffle=True, random_state=s.skfseed)
        all_keep_indices = []
        for foldnum, (dont_use_index, keep_index) in enumerate(skf1):
            if foldnum*10 < s.percentageofinstances:
                print('yup',keep_index)
                # np.append(all_keep_indices,keep_index)
                all_keep_indices+=[item for item in keep_index]
                print(len(all_keep_indices))
        train_data=train_data[all_keep_indices,:]
        train_labels=train_labels[all_keep_indices]
        print(train_labels)
    print('\n shapes after:', train_data.shape, train_labels.shape, test_data.shape, test_labels.shape)
    # sys.exit()
    return np.asarray(train_data), np.asarray(test_data), np.asarray(train_labels), np.asarray(test_labels)#, train_indices, test_indices

# def get_awa_data(dataset_index):
#     data = np.load(get_full_path('Desktop/Privileged_Data/data_Joe/data{}.npy'.format(dataset_index)))
#     # print (np.sum(labels>0))
#     print ('data shape',data.shape)
#     # print (data[0])
#     # print (labels.shape)
#     pos_instances,neg_instances = data[:252], data[252:]
#     print (pos_instances.shape, neg_instances.shape)
#     return(pos_instances,neg_instances)

def get_awa_data(dataset_index):
    data = np.load(get_full_path('Desktop/Privileged_Data/data_easyhard_Joe_allsamples/data{}.npy'.format(dataset_index)))
    labels = np.load(get_full_path('Desktop/Privileged_Data/data_easyhard_Joe_allsamples/labels{}.npy'.format(dataset_index)))
    pos_instances,neg_instances = data[labels==1.], data[labels==-1.]
    print (pos_instances.shape, neg_instances.shape)
    return(pos_instances,neg_instances)



######## BELOW CODE WAS USED TO SAVE DATA TO NUMPY ARRAYS FOR POS AND NEG


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
    # print (positive_instances.shape, negative_instances.shape)

    return(positive_instances,negative_instances)


def get_tech_address(dataset_index):
    with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_295.txt"), "r") as infile:
    # with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_49.txt"), "r") as infile:
        line = infile.readlines()[dataset_index]
        return (get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/{}/vectors.dat".format(line.strip('\r\n'))))


def get_words(dataset_index):
    with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_295.txt"), "r") as infile:
    # with open(get_full_path("Desktop/Privileged_Data/techtc300_preprocessed/techtc_files_49.txt"), "r") as infile:
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


# Call the following once, to save pos and negative instances for each of 295 datasets

# for datasetnum in range(295):
#     pos_instances, neg_instances = get_techtc_data(datasetnum)
#     np.save(get_full_path("Desktop/Privileged_Data/SavedDatasets/tech/{}positive".format(datasetnum)), pos_instances)
#     np.save(get_full_path("Desktop/Privileged_Data/SavedDatasets/tech/{}negative".format(datasetnum)), neg_instances)



# Call the following once, to save pos and negative instances for each of 295 datasets

def save_train_test_indices(dataset, datasetnum):
    folder = get_full_path('Desktop/Privileged_Data/SavedTrainTestIndices/{}{}'.format(dataset,datasetnum))
    if pc_instances!=100:
        folder = get_full_path('Desktop/Privileged_Data/SavedTrainTestIndices/{}{}-{}pc'.format(dataset, datasetnum, pc_instances))
    print(folder)
    if not(os.path.exists(folder)):
        mkdir(folder)
    class0_data, class1_data = load_dataset_from_name(dataset,datasetnum)
    all_labels = np.r_[[-1]*class0_data.shape[0], [1]* class1_data.shape[0]]
    for skfseed in range(1,2):
        skf = StratifiedKFold(all_labels, n_folds=10, shuffle=True, random_state=skfseed)
        for foldnum, (train_index, test_index) in enumerate(skf):
            np.random.seed(skfseed)
            orig_num_train_instances = len(train_index)
            print('orig',orig_num_train_instances)
            num_of_train_instances = orig_num_train_instances * pc_instances // 100
            indices = np.random.choice(orig_num_train_instances, num_of_train_instances, replace=False)
            print('old', len(train_index))
            train_index = train_index.copy()[indices]
            print('new', len(train_index))
            np.save(os.path.join(folder,'{}-{}-train'.format(skfseed,foldnum)),train_index)
            np.save(os.path.join(folder,'{}-{}-test'.format(skfseed, foldnum)),test_index)


# for datasetnum in range(295):
#     for pc_instances in range(10,100,10):
#         save_train_test_indices('tech',datasetnum,pc_instances)

# for dataset in ['arcene','madelon','gisette','dexter','dorothea']:
#     save_train_test_indices(dataset, 0)

# save_train_test_indices('bbc',0)
