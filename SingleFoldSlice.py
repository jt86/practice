'''
This is the main function.
Things to check before running: (1) values of C, (2) output directory and whether old output is there
(3) number of jobs in go-practice-submit.sh matches desired number of settings to run in Run Experiment
(4) that there is no code test run
(5) data is regularised as desired in GetSingleFoldData
(6) params including number of folds and stepsize set correctly
'''

import os

# print os.environ['HOME']

import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus import svmplusQP, svmplusQP_Predict
from ParamEstimation import get_best_params, get_best_CandCstar, get_best_params_dp2
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold
import sys
import numpy.random
from sklearn import preprocessing
# from time import time
from sklearn.metrics import accuracy_score
import numpy as np
from New import svm_problem, svm_u_problem
from Models import SVMdp, SVMu, get_accuracy_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
import socket


# from sklearn.feature_selection import mutual_info_classif
from pprint import pprint
# print (PYTHONPATH)


############## HELPER FUNCTIONS

def get_priv_subset(feature_set,take_top_t,percent_of_priv):
    num_of_priv_feats = percent_of_priv * feature_set.shape[1] // 100
    assert take_top_t in ['top','bottom']
    if take_top_t == 'top':
        priv_train = feature_set[:, :num_of_priv_feats]
    if take_top_t == 'bottom':
        priv_train = feature_set[:, -num_of_priv_feats:]
    print('privileged data shape', priv_train.shape)
    return priv_train

def make_directory(directory):
    try:
        os.makedirs(directory)
    except OSError:
        if not os.path.isdir(directory):
            raise
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_random_array(num_instances, num_feats):
    random_array = np.random.rand(num_instances, num_feats)
    random_array = preprocessing.scale(random_array)
    return random_array

def take_subset(all_training, training_labels, percentageofinstances):
    orig_num_train_instances = all_training.shape[0]
    num_of_train_instances = orig_num_train_instances * percentageofinstances // 100
    indices = np.random.choice(orig_num_train_instances, num_of_train_instances, replace=False)
    all_training = all_training.copy()[indices, :]
    training_labels = training_labels[indices]
    print(all_training.shape)
    print(training_labels.shape)
    print(indices)

def save_scores(s, score, cross_val_folder):
    print('{} score = {}'.format(s.classifier, score))
    with open(os.path.join(cross_val_folder, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'a') as cv_lupi_file:
        cv_lupi_file.write('{},{}'.format(s.foldnum,score) + '\n')


############## LUFe FUNCTIONS

def delta_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    C, gamma, delta = get_best_params_dp2(s, normal_train, labels_train, priv_train, cross_val_folder)
    problem = svm_problem(normal_train, priv_train, labels_train, C=C,
                          gamma=gamma, delta=delta)
    s2 = SVMdp()
    dp_classifier = s2.train(prob=problem)
    dp_score = get_accuracy_score(dp_classifier, normal_test, labels_test)
    save_scores(s,dp_score,cross_val_folder)


def svm_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    c_svm_plus, c_star_svm_plus = get_best_CandCstar(s, normal_train, labels_train, priv_train, cross_val_folder)
    duals, bias = svmplusQP(normal_train, labels_train.copy(), priv_train, c_svm_plus, c_star_svm_plus)
    lupi_predictions = svmplusQP_Predict(normal_train, normal_test, duals, bias).flatten()
    accuracy_lupi = np.sum(labels_test == np.sign(lupi_predictions)) / (1. * len(labels_test))
    save_scores(s, accuracy_lupi, cross_val_folder)

def do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    # priv_train = get_priv_subset(priv_train, s.take_top_t, s.percent_of_priv)
    if s.lupimethod == 'dp':
        delta_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder)
    if s.lupimethod == 'svmplus':
        svm_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder)

############## FUNCTIONS TO GET SUBSETS OF FEATURES AND SUBSETS OF INSTANCES


def do_rfe(s, all_train, all_test, labels_train, labels_test,cross_val_folder):
    best_rfe_param = get_best_params(s, all_train, labels_train, cross_val_folder, 'rfe')
    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.foldnum)
    rfe = RFE(estimator=svc, n_features_to_select=s.topk, step=s.stepsize)
    rfe.fit(all_train, labels_train)
    make_directory(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/'.format(s.topk, s.featsel)))
    np.save(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/{}{}-{}-{}'.
                          format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.foldnum)), (np.argsort(rfe.ranking_)))
    # print ('rfe ranking',rfe.ranking_)
    # print(np.count_nonzero(rfe.ranking_ == 1))
    # print(np.argsort(rfe.ranking_))
    # print(rfe.ranking_[np.argsort(rfe.ranking_)])
    support = np.where(rfe.support_ == True)
    ranking = np.where(rfe.ranking_ == 1)
    print(len(support),len(ranking))
    print('support',support)
    print ('ranking',ranking)
    print ('are they equal?',np.array_equal(support,ranking))
    do_svm_for_rfe(s, all_train, all_test, labels_train, labels_test, cross_val_folder, best_rfe_param,support)


def do_svm_for_rfe(s,all_train,all_test,labels_train,labels_test,cross_val_folder, best_rfe_param,support):
    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s,all_train,all_test,support)
    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.foldnum)
    svc.fit(normal_train, labels_train)
    rfe_accuracy = svc.score(normal_test, labels_test)
    save_scores(s, rfe_accuracy, cross_val_folder)


# def get_norm_priv(s,all_train,all_test):
#     if s.featsel == 'rfe':
#         best_n_mask = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/{}{}-{}-{}.npy'.
#                               format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
#         normal_train = all_train[:, best_n_mask].copy()
#         normal_test = all_test[:, best_n_mask].copy()
#         priv_train = all_train[:, np.invert(best_n_mask)].copy()
#         priv_test = all_test[:, np.invert(best_n_mask)].copy()
#         # all_features_ranking = rfe.ranking_[np.invert(best_n_mask)] # gets just the unselected features' rankings
#         # priv_train = priv_train[:, np.argsort(all_features_ranking)] #reorder train and test using this
#         # priv_test = priv_test[:, np.argsort(all_features_ranking)]
#         return normal_train, normal_test, priv_train, priv_test


def get_norm_priv(s,all_train,all_test):
    ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/{}{}-{}-{}.npy'.
                          format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
    # assert(np.array_equal(support[0], np.sort(ordered_indices[:s.topk])))
    sorted_training = all_train[:, ordered_indices]
    sorted_testing = all_test[:, ordered_indices]

    normal_train = sorted_training[:, :s.topk]  # take the first n_top_feats
    normal_test = sorted_testing[:, :s.topk]
    priv_train = sorted_training[:, s.topk:]
    priv_test = sorted_testing[:, s.topk:]
    print('nor tr',normal_train.shape,'nor te',normal_test.shape,'pri tr',priv_train.shape,'pri te',priv_test.shape)
    return normal_train, normal_test, priv_train, priv_test

def do_mutinfo(s, all_train, labels_train, all_test,labels_test,cross_val_folder):
    scores = mutual_info_classif(all_train, labels_train)
    ordered_indices = np.array(np.argsort(scores)[::-1])  # ordered feats is np array of indices from biggest to smallest
    make_directory(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/'.format(s.topk, s.featsel)))
    np.save(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/{}{}-{}-{}'.
                          format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.foldnum)), ordered_indices)

    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s,all_train,all_test)
    do_svm(s, normal_train, labels_train, normal_test, labels_test, cross_val_folder)


def do_svm(s, train_data, labels_train, test_data, labels_test, cross_val_folder):
    best_C = get_best_params(s, train_data, labels_train, cross_val_folder, 'svm')
    clf = svm.SVC(C=best_C, kernel=s.kernel, random_state=s.foldnum)
    print('labels data', labels_train.shape)
    clf.fit(train_data, labels_train)
    predictions = clf.predict(test_data)
    score = accuracy_score(labels_test, predictions)
    save_scores(s, score, cross_val_folder)




##################################################################################################

def single_fold(s):
    print(s.cvalues)
    pprint(vars(s))
    print('{}% of train instances; {}% of discarded feats used as priv'.format(s.percentageofinstances,s.percent_of_priv))
    np.random.seed(s.foldnum)
    output_directory = get_full_path((
                                     'Desktop/Privileged_Data/MayResults/{}-{}-{}-{}selected-{}{}priv/{}{}/').format(
        s.classifier,s.lupimethod, s.featsel, s.topk, s.take_top_t,s.percent_of_priv,s.dataset, s.datasetnum))
    make_directory(output_directory)

    all_train, all_test, labels_train, labels_test = get_train_and_test_this_fold(s)
    if s.classifier == 'baseline':
        do_svm(s, all_train, labels_train, all_test, labels_test, output_directory)
    elif s.classifier == 'featselector':
        if s.featsel == 'rfe':
            print('doing rfe')
            do_rfe(s,all_train,all_test,labels_train,labels_test,output_directory)
        if s.featsel == 'mi':
            do_mutinfo(s, all_train, labels_train, all_test,labels_test, output_directory)
    else:
        normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)
        if s.classifier == 'lufe':
            do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, output_directory)
        if s.classifier == 'lufereverse':
            do_lufe(s, priv_train, labels_train, normal_train, priv_test, labels_test, output_directory)
        if s.classifier == 'svmreverse':
            do_svm(s, priv_train, labels_train, priv_test, labels_test, output_directory)

##################################################################################################



class Experiment_Setting:
    def __init__(self, foldnum, topk, dataset, datasetnum, kernel, cmin, cmax, numberofcs, skfseed,
                 percent_of_priv, percentageofinstances, take_top_t, lupimethod, featsel, classifier):

        assert classifier in ['baseline','featselector','lufe','lufereverse','svmreverse']
        assert lupimethod in ['nolufe','svmplus','dp'], 'lupi method must be nolufe, svmplus or dp'
        assert featsel in ['nofeatsel','rfe','mi','anova'], 'feat selection method must be nofeatsel, rfe, mi or anova'

        self.foldnum = foldnum
        self.topk = topk
        self.dataset = dataset
        self.datasetnum = datasetnum
        self.kernel = kernel
        # self.cvalues = np.logspace(*[int(item)for item in cvalues.split('a')])
        self.cvalues = (np.logspace(cmin,cmax,numberofcs))
        self.skfseed = skfseed
        self.percent_of_priv = percent_of_priv
        self.percentageofinstances = percentageofinstances
        self.take_top_t = take_top_t
        self.lupimethod =lupimethod
        self.stepsize=0.1
        self.featsel = featsel
        self.classifier = classifier



        if self.classifier == 'baseline':
            self.lupimethod='nolufe'
            self.featsel='nofeatsel'
            self.topk='all'
        if self.classifier == 'featselector':
            self.lupimethod='nolufe'

    def print_all_settings(self):
        pprint(vars(self))
        # print(self.k,self.top)

#
# for datasetnum in range(280,295):
#     for i in range(10):
#         setting = Experiment_Setting(foldnum=i, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=0,
#                                      percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='nolufe', featsel='mi', classifier='featselector')
#         setting.print_all_settings()
#         single_fold(setting)

# data = (np.load('/Volumes/LocalDataHD/j/jt/jt306/Desktop/SavedIndices/top300RFE/tech0-0-0.npy'))
# # cvalues = '-3a3a7'
# # print([int(item)for item in cvalues.split('a')])
# seed = 1
# dataset='tech'
# top_k = 30
# take_top_t ='top'
# percentofpriv = 100
#
# setting = Experiment_Setting(foldnum=7, topk=300, dataset='tech', datasetnum=209, kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=1,
#                                  percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='lufe')
# single_fold(setting)


#
# for foldnum in range(10):
#     for lupimethod in ['dp','svmplus']:
#         for featsel in ['rfe']:
#             for datasetnum in range (259,295): #5
#                 setting = Experiment_Setting(k=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
#                          cmin=-3,cmax=3,numberofcs=7, skfseed=0, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod=lupimethod,
#                          featsel=featsel)
#                 break
#                 # single_fold(setting)
