from GetSingleFoldData import get_train_and_test_this_fold
from SingleFoldSlice import Experiment_Setting,get_norm_priv
from Get_Full_Path import get_full_path
import numpy as np
import scipy.io as sio

dataset='tech'
datasetnum = 1

def save_for_matlab(s,num_instances='all',num_feats='all'):

    train_data, test_data,train_labels, test_labels =get_train_and_test_this_fold(s)
    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s,train_data,test_data)
    # if num_feats != 'all':
    #
    # normal_train = normal_train[:num_instances,:num_feats]
    # normal_test = normal_test[:num_instances, :num_feats]
    # priv_train = priv_train[:num_instances, :num_feats]
    # priv_test = priv_test[:num_instances, :num_feats]


    # normal_train = normal_train[:,:num_feats]
    # normal_test = normal_test[:,:num_feats]
    # priv_train = priv_train[:, :num_feats]
    # priv_test = priv_test[:, :num_feats]

    # train_labels=train_labels[:num_instances]
    # test_labels = test_labels[:num_instances]


    # if num_instances != 'all':
    #     for item in [normal_train, normal_test, priv_train, priv_test]:
    #         item = item[:num_instances,:]

    new_normal_train = np.hstack((normal_train,np.zeros((normal_train.shape[0],priv_train.shape[1]))))
    new_priv_train = np.hstack((np.zeros((priv_train.shape[0],normal_train.shape[1])),priv_train))

    new_normal_test = np.hstack((normal_test, np.zeros((normal_test.shape[0], priv_test.shape[1]))))
    new_priv_test = np.hstack((np.zeros((priv_test.shape[0], normal_test.shape[1])), priv_test))

    all_train_labels = np.hstack((train_labels,train_labels))
    all_test_labels = np.hstack((test_labels,test_labels))
    print('labels',all_train_labels.shape,all_test_labels.shape,)

    all_train = np.vstack((new_normal_train,new_priv_train))
    all_test = np.vstack((new_normal_test,new_priv_test))

    print('data', all_train.shape,all_test.shape)
    sio.savemat(get_full_path('Desktop/Privileged_Data/Matlab_Datasets/{}-train-x.mat'.format(s.name)), {'vect': all_train})
    sio.savemat(get_full_path('Desktop/Privileged_Data/Matlab_Datasets/{}-train-y.mat'.format(s.name)), {'vect': all_train_labels})
    sio.savemat(get_full_path('Desktop/Privileged_Data/Matlab_Datasets/{}-test-x.mat'.format(s.name)), {'vect': all_test})
    sio.savemat(get_full_path('Desktop/Privileged_Data/Matlab_Datasets/{}-test-y.mat'.format(s.name)), {'vect': all_test_labels})


for foldnum in range(10):
    s = Experiment_Setting(foldnum=1, topk=300, dataset='tech', datasetnum=0, kernel='linear',
                           cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                           take_top_t='top', lupimethod='svmplus',
                           featsel='rfe', classifier='lufe', stepsize=0.1)

    save_for_matlab(s,'all',num_feats=300)