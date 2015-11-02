__author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
import csv
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, Imputer, Normalizer
from scipy import sparse as sp
import sys
from sklearn.cross_validation import StratifiedKFold

def get_train_and_test_this_fold(dataset,datasetnum,k, skf_seed):	#N,test_N per class
    if dataset == 'synthetic':
        class0_data, class1_data = np.loadtxt(get_full_path('Desktop/Privileged_Data/syn_positive.txt')), np.loadtxt(get_full_path('Desktop/Privileged_Data/syn_negative.txt'))

    if dataset=='arcene':
        class0_data, class1_data = get_arcene_data()
        # N, test_N =14,30
    if dataset=='madelon':
        class0_data, class1_data = get_madelon_data()
        # N, test_N = 250,250
    if dataset=='gisette':
        class0_data, class1_data = get_gisette_data()
        # N, test_N = 100,200
    if dataset=='dexter':
        class0_data, class1_data = get_dexter_data()
        # N, test_N = 50,100
    if dataset=='dorothea':
        class0_data, class1_data = get_dorothea_data()
        # N, test_N = 26,52
    if dataset=='mushroom':
        class0_data, class1_data = get_mushroom_data()
        N, test_N = 500,1000
    if dataset=='sick':
        class0_data, class1_data = get_sick_data()
        N, test_N = 77,154
    if 'prove' in dataset:
        id = dataset[-1]
        print('id is ',id)
        class0_data,class1_data=get_prove_data(id)
        N, test_N = 79, 150
    if dataset=='heart':
        class0_data,class1_data=get_heart_data()
        N, test_N = 40,80
    if dataset=='vote':
        class0_data,class1_data=get_vote_data()
        N, test_N = 112,56
    # print class0_data.shapef, class1_data.shape

    if dataset=='tech':
        class0_data,class1_data=get_techtc_data(datasetnum)
    class0_labels = [-1]*class0_data.shape[0]
    class1_labels = [1]* class1_data.shape[0]
    all_labels = np.r_[class0_labels, class1_labels]
    print (all_labels.shape)


    all_data = np.vstack([class0_data,class1_data])
    print(all_data.shape)
    # sys.exit()

    # test_N = min(class0_data.shape[0],class1_data.shape[0])//3
    # N = test_N*2
    # print (N, test_N)


    # idx1 = np.random.permutation(class0_data.shape[0])
    # train1, test1 = idx1[:N], idx1[N:N+test_N]
    # idx2 = np.random.permutation(class1_data.shape[0])
    # train2, test2 = idx2[:N], idx2[N:N+test_N]
   #
    # train_data = np.r_[class0_data[train1], class1_data[train2]]
    # test_data = np.r_[class0_data[test1], class1_data[test2]]
    # train_labels = np.ravel(np.r_[[1]*N, [-1]*N])
    # test_labels = np.ravel(np.r_[[1]*test_N, [-1]*test_N])
    #
    # print('train', train_labels.shape, train_labels)
    # print('test', test_labels.shape, test_labels)
    # print ('all', all_labels.shape, all_labels)


    skf = StratifiedKFold(all_labels, n_folds=4, random_state=skf_seed)
    for fold_num, (train_index, test_index) in enumerate(skf):
        if fold_num==k:
            train_data, test_data = all_data[train_index],all_data[test_index]
            train_labels, test_labels = all_labels[train_index], all_labels[test_index]


    #L1 normalization ============================
    normaliser = Normalizer(norm='l1')
    np.set_printoptions(threshold=np.nan)
    train_data=normaliser.fit_transform(train_data)

    normaliser = Normalizer(norm='l1')
    test_data=normaliser.fit_transform(test_data)
    # train_data = train_data/np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, train_data).reshape(-1,1)
    # test_data = test_data/np.apply_along_axis(lambda row:np.linalg.norm(row,ord=1), 1, test_data).reshape(-1,1)
    print ('train data shape', train_data.shape, 'test data shape', test_data.shape)
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
        print (labels_array.shape)
        print (labels_array)
        for row_num, line in enumerate(file):
            print (row_num,line)
            labels_array[row_num] = line

        # print (labels_array)
        # print (labels_array.shape)

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

def get_mushroom_data():

    numerical_value_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15,
        'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26 ,'?':27}


    with open(get_full_path('Desktop/Privileged_Data/Mushroom/mushroom.data.csv'), "r+") as infile:
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
        # print item

    print(features_array[0])
    # dummies = get_dummies(features_array)

    print(features_array.shape, labels_array.shape)
    print(labels_array)

    enc = OneHotEncoder(n_values='auto', categorical_features ='all')
    features_array = enc.fit_transform(features_array)
    # print features_array[0]
    print('feats array shape',features_array.shape)

    positive_instances = (features_array[labels_array==1]).todense()
    negative_instances = (features_array[labels_array==-1]).todense()

    print(positive_instances.shape, negative_instances.shape)
    return positive_instances, negative_instances


def get_sick_data():
    print('Reading Sick data from disk')
    with open(get_full_path("Desktop/Privileged_Data/Sick/sick.data.txt"), "r+") as infile:
        features_array = np.loadtxt(infile, dtype=str)
        features_array.shape = (3772, 30)
    labels_array=np.zeros(3772)
    for index,item in enumerate(features_array[:,-1]):
        if 'sick' in item:
            labels_array[index]=1
        else:
            labels_array[index]=-1

    with open(get_full_path("Desktop/Privileged_Data/Sick/sick2.txt"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=float,delimiter="\t", filling_values=np.NaN)
        features_array.shape = (3772, 29)

    final_feat=features_array[:,-1]
    final_feat.shape=(3772,1)
    features_array = features_array[:,:-1]

    one_hot_array = np.zeros((3772,5))
    for index,feature in enumerate(final_feat):
        one_hot_array[index,int(feature)-1]=1

    imp = Imputer(missing_values=np.NaN, strategy='mean', axis=0)
    features_array=imp.fit_transform(features_array)

    features_array=np.hstack((features_array,one_hot_array))
    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    print(positive_instances.shape, negative_instances.shape)
    return positive_instances, negative_instances

def get_prove_data(label_number):
    print('Reading Prove data from disk')
    with open(get_full_path("Desktop/Privileged_Data/ml-prove/train.csv"), "r+") as infile:
        features_array = np.loadtxt(infile,delimiter=',', dtype=float)
    print(len(features_array[0]))
    print(len(features_array))


    all_labels = features_array[:,-6:]
    features_array=features_array[:,:-6]
    print(features_array.shape)

    labels_array = all_labels[:,label_number]
    print(labels_array)

    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances

def get_heart_data(debug=False):
    print('Reading HEART data from disk')

    with open(get_full_path("Desktop/Privileged_Data/new_data/heart.dat"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=str)
        features_array = np.array(str(features_array).translate(None, '()').split(","), dtype=float)
        features_array.shape = (270, 14)
        labels_array = np.array(features_array[:, 13])
        features_array = np.array(features_array[:, :13])
        labels_array[labels_array == 2] = -1


    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances

def get_vote_data():
    print('Reading VOTE data from disk')
    with open(get_full_path("Desktop/Privileged_Data/new_data/house-votes-84.data"), "r+") as infile:
        features_array = np.genfromtxt(infile, dtype=None)

        features_array.shape = (435, 17)

        d = {'republican':1.,'democrat':-1., 'y':1., 'n':0., '?':0.5}
        new_array = np.copy(features_array)
        for k, v in d.items():
            new_array[features_array==k] = v


    labels_array = np.array(new_array[:,0], dtype=float)
    features_array = np.array(new_array[:,1:], dtype=float)
    positive_instances = (features_array[labels_array==1])
    negative_instances = (features_array[labels_array==-1])
    return positive_instances, negative_instances




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