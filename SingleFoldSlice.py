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
sys.path.append('bahsic')
# print(sys.path)
from bahsic import CBAHSIC,vector


# from sklearn.feature_selection import mutual_info_classif
from pprint import pprint
# print (PYTHONPATH)


############## HELPER FUNCTIONS
#get_priv_subset(priv_train, s.take_top_t, s.percent_of_priv)
def get_priv_subset(feature_set,take_top_t,percent_of_priv):
    print('taking privileged subset')
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

def svm_u(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    C, gamma, delta = get_best_params_dp2(s, normal_train, labels_train, priv_train, cross_val_folder)

    problem = svm_u_problem(X=normal_train, Xstar=priv_train,XstarStar=priv_train, Y=labels_train, C=C,
                          gamma=gamma, delta=delta)
    # def __init__(self, X, Xstar, XstarStar, Y, C=1.0, gamma=1.0, sigma=1, delta=1.0, xkernel=Linear(),
    #              xSkernel=Linear(), xSSkernel=Linear()):
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
    priv_train = get_priv_subset(priv_train, s.take_top_t, s.percent_of_priv)
    if s.lupimethod == 'dp':
        delta_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder)
    if s.lupimethod == 'svmplus':
        svm_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder)
    if s.lupimethod == 'dsvm':
        do_dsvm(s, normal_train, labels_train,  priv_train, normal_test, labels_test, cross_val_folder)
    if s.lupimethod == 'svm_u':
        do_dsvm(s, normal_train, labels_train,  priv_train, normal_test, labels_test, cross_val_folder)

def do_luferandom(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    num_instances,num_feats = priv_train.shape
    np.random.seed(s.foldnum)
    random_array = np.random.rand(num_instances, num_feats)
    print('random \n',random_array)
    random_priv_train = preprocessing.scale(random_array)
    do_lufe(s, normal_train, labels_train, random_priv_train, normal_test, labels_test, cross_val_folder)

def do_lufeshuffle(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    np.random.seed(s.foldnum)
    print('\n',priv_train)
    np.random.shuffle(priv_train)
    print('\n', priv_train)
    do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder)

def do_lufetrain(s, normal_train, labels_train, priv_train, cross_val_folder):
    if s.lupimethod == 'svmplus':
        svm_plus(s, normal_train, labels_train, priv_train, normal_train, labels_train, cross_val_folder)

############## FUNCTIONS TO GET SUBSETS OF FEATURES AND SUBSETS OF INSTANCES




def do_rfe(s, all_train, all_test, labels_train, labels_test,cross_val_folder):
    best_rfe_param = get_best_params(s, all_train, labels_train, cross_val_folder, 'rfe')
    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.foldnum)
    rfe = RFE(estimator=svc, n_features_to_select=s.topk, step=0.1)
    print(all_train.shape,labels_train.shape)
    rfe.fit(all_train, labels_train)
    stepsize = str(s.stepsize).replace('.', '-')
    # directory = get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-{}instances/'.format(s.topk, s.featsel,s.percentageofinstances))
    directory = get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/'.format(s.topk, s.featsel,s.percentageofinstances))
    make_directory(directory)
    np.save(get_full_path(directory+'/{}{}-{}-{}'.format(s.dataset, s.datasetnum, s.skfseed, s.foldnum)), (np.argsort(rfe.ranking_)))
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
    do_svm_for_rfe(s, all_train, all_test, labels_train, labels_test, cross_val_folder, best_rfe_param)


def do_svm_for_rfe(s,all_train,all_test,labels_train,labels_test,cross_val_folder, best_rfe_param):
    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s,all_train,all_test)
    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.foldnum)
    svc.fit(normal_train, labels_train)
    rfe_accuracy = svc.score(normal_test, labels_test)
    save_scores(s, rfe_accuracy, cross_val_folder)


def do_dsvm(s, normal_train, labels_train,  priv_train, normal_test, labels_test, folder):
    make_directory(os.path.join(folder,'dvalues'))
    c = get_best_params(s, priv_train, labels_train, folder, 'svm')
    svc = SVC(C=c, kernel=s.kernel, random_state=s.foldnum)
    svc.fit(priv_train, labels_train)
    d_i = np.array([1 - (labels_train[i] * svc.decision_function(priv_train)[i]) for i in range(len(labels_train))])
    print(d_i.shape)
    d_i = np.reshape(d_i, (d_i.shape[0], 1))
    np.save(os.path.join(folder,'dvalues/{}{}-{}-{}'.format(s.dataset, s.datasetnum, s.skfseed, s.foldnum)),d_i)
    svm_plus(s, normal_train, labels_train, d_i, normal_test, labels_test, folder)


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
    '''
    :param s: experiment setting
    :param all_train:
    :param all_test:
    :return: 4 np arrays: the dataset split into training/testing and normal/privileged
    '''
    if s.featsel=='rfe' and s.dataset=='tech' and s.percentageofinstances==100:
        stepsize = str(s.stepsize).replace('.', '-')
        ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-step{}/{}{}-{}-{}.npy'.
                              format(s.topk, s.featsel, stepsize,s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
    elif s.featsel == 'rfe' and s.dataset == 'tech' and s.percentageofinstances!=100:
        stepsize = str(s.stepsize).replace('.', '-')
        ordered_indices = np.load(
            get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-step{}-{}instances/{}{}-{}-{}.npy'.
                          format(s.topk, s.featsel, stepsize, s.percentageofinstances,s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
    elif s.percentageofinstances != 100:
        print('not 100')
        ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-{}instances/{}{}-{}-{}.npy'.
                                                format(s.topk, s.featsel, s.percentageofinstances, s.dataset, s.datasetnum, s.skfseed,
                                                       s.foldnum)))
    else:
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

    save_indices_do_svm(s, all_train, all_test, labels_train, labels_test, cross_val_folder, ordered_indices)


def do_anova(s, all_train, all_test, labels_train, labels_test, output_directory):
    # get array of scores in the same order as original features
    selector = SelectPercentile(f_classif, percentile=100)
    selector.fit(all_train, labels_train)
    scores = selector.scores_
    # sort the scores (small to big)
    sorted_indices_small_to_big = np.argsort(scores)
    sorted_scores = scores[sorted_indices_small_to_big]
    # nan indices are at the end. Reverse the score array (big to small) but leave these at the end
    nan_indices = np.argwhere(np.isnan(scores)).flatten()
    non_nan_indices = sorted_indices_small_to_big[:-len(nan_indices)]
    sorted_indices_big_to_small = non_nan_indices[::-1]
    ordered_indices = np.array(np.hstack([sorted_indices_big_to_small, nan_indices]))  # indices of biggest

    print(scores[ordered_indices])
    print('ordered feats', ordered_indices.shape)

    save_indices_do_svm(s, all_train, all_test, labels_train, labels_test, output_directory, ordered_indices)


def save_indices_do_svm(s, all_train, all_test, labels_train, labels_test, output_directory, ordered_indices):
    if s.percentageofinstances==100:
        directory = (get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/'.format(s.topk, s.featsel)))
    else:
        directory = (get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-{}instances/'.format(s.topk, s.featsel, s.percentageofinstances)))
    make_directory(directory)
    np.save(directory+'/{}{}-{}-{}'.format(s.dataset, s.datasetnum, s.skfseed, s.foldnum), ordered_indices)

    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)
    do_svm(s, normal_train, labels_train, normal_test, labels_test, output_directory)

def do_chi2(s, all_train, all_test, labels_train, labels_test, output_directory):
    # add values so that all features of training data are non-zero
    all_training=all_train-np.min(all_train)
    # get array of scores in the same order as original features
    selector = SelectPercentile(chi2, percentile=100)
    selector.fit(all_training, labels_train)
    scores = selector.scores_
    # sort the scores (small to big)
    ordered_indices = np.argsort(scores)[::-1]
    print(scores[ordered_indices])
    print(ordered_indices)

    save_indices_do_svm(s, all_train, all_test, labels_train, labels_test, output_directory, ordered_indices)


def do_bahsic(s, all_train, all_test, labels_train, labels_test, output_directory):
    cbahsic = CBAHSIC()
    # output2 =((cbahsic.BAHSICOpt(x=x, y=y, kernelx=vector.CLinearKernel(), kernely=vector.CLinearKernel(), flg3=50, flg4=0.5)))
    labels_train=(labels_train.reshape(len(labels_train),1))
    ordered_indices = cbahsic.BAHSICOpt(x=all_train, y=labels_train, kernelx=vector.CLinearKernel(), kernely=vector.CLinearKernel(), flg3=s.topk, flg4=s.stepsize)
    print(ordered_indices)
    ordered_indices=ordered_indices[::-1]
    print(ordered_indices)
    labels_train = np.ndarray.flatten(labels_train)
    save_indices_do_svm(s, all_train, all_test, labels_train, labels_test, output_directory, ordered_indices)

def do_svm(s, train_data, labels_train, test_data, labels_test, cross_val_folder):
    print ('train',train_data.shape,'labels',labels_train.shape,'test',test_data.shape,'labels',labels_test.shape)
    best_C = get_best_params(s, train_data, labels_train, cross_val_folder, 'svm')
    clf = svm.SVC(C=best_C, kernel=s.kernel, random_state=s.foldnum)
    print('labels data', labels_train.shape)
    clf.fit(train_data, labels_train)
    predictions = clf.predict(test_data)
    score = accuracy_score(labels_test, predictions)
    save_scores(s, score, cross_val_folder)

def do_svm_on_train(s, all_train, all_test, labels_train, cross_val_folder):

    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)

    do_svm(s, normal_train, labels_train, normal_train, labels_train, cross_val_folder)



def take_subset(s,all_train, labels_train, percentageofinstances):
    print(all_train.shape)
    orig_num_train_instances = all_train.shape[0]
    num_of_train_instances = orig_num_train_instances * percentageofinstances // 100
    np.random.seed(s.foldnum)
    indices = np.random.choice(orig_num_train_instances, num_of_train_instances, replace=False)
    all_train = all_train.copy()[indices, :]
    labels_train = labels_train[indices]
    print(all_train.shape)
    print(labels_train.shape)
    print(indices)
    return all_train, labels_train
##################################################################################################

def single_fold(s):
    print(s.cvalues)
    pprint(vars(s))
    print('{}% of train instances; {}% of discarded feats used as priv'.format(s.percentageofinstances,s.percent_of_priv))
    np.random.seed(s.foldnum)
    output_directory = get_full_path(('Desktop/Privileged_Data/NIPSResults/{}/{}{}/').format(s.name,s.dataset, s.datasetnum))

    make_directory(output_directory)

    all_train, all_test, labels_train, labels_test = get_train_and_test_this_fold(s)
    all_train, labels_train = take_subset(s, all_train, labels_train, s.percentageofinstances)

    if s.dataset != 'tech':
        s.topk = (all_train.shape[1]*s.topk)//100
    print('all train',all_train.shape[1])
    print('topk',s.topk)
    # sys.exit()


    # all_train=all_train[:,:500]
    # all_test=all_test[:,:500]
    if s.classifier == 'baseline':
        do_svm(s, all_train, labels_train, all_test, labels_test, output_directory)
    elif s.classifier == 'featselector':
        if s.featsel == 'rfe':
            print('doing rfe')
            do_rfe(s,all_train,all_test,labels_train,labels_test,output_directory)
        if s.featsel == 'mi':
            do_mutinfo(s, all_train, labels_train, all_test,labels_test, output_directory)
        if s.featsel == 'anova':
            do_anova(s, all_train, all_test, labels_train, labels_test, output_directory)
        if s.featsel == 'chi2':
            do_chi2(s, all_train, all_test, labels_train, labels_test, output_directory)
        if s.featsel == 'bahsic':
            do_bahsic(s, all_train, all_test, labels_train, labels_test, output_directory)
    else:
        normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)
        if s.classifier == 'lufe':
            do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, output_directory)
        if s.classifier == 'lufereverse':
            do_lufe(s, priv_train, labels_train, normal_train, priv_test, labels_test, output_directory)
        if s.classifier == 'svmreverse':
            do_svm(s, priv_train, labels_train, priv_test, labels_test, output_directory)
        if s.classifier == 'luferandom':
            do_luferandom(s, normal_train, labels_train, priv_train, normal_test, labels_test, output_directory)
        if s.classifier == 'lufeshuffle':
            do_lufeshuffle(s, normal_train, labels_train, priv_train, normal_test, labels_test, output_directory)
        if s.classifier == 'svmtrain':
            do_svm_on_train(s, all_train, all_test, labels_train, output_directory)
        if s.classifier == 'lufetrain':
            do_lufetrain(s, normal_train, labels_train, priv_train, output_directory)
##################################################################################################



class Experiment_Setting:
    def __init__(self, classifier, datasetnum, lupimethod, featsel, stepsize=0.1, foldnum='all', topk=300, dataset='tech', skfseed=1, kernel='linear',
                 cmin=-3, cmax=3, numberofcs=7, percent_of_priv=100, percentageofinstances=100, take_top_t='top'):

        assert classifier in ['baseline','featselector','lufe','lufereverse','svmreverse','luferandom','lufeshuffle', 'svmtrain','lufetrain']
        assert lupimethod in ['nolufe','svmplus','dp','dsvm'], 'lupi method must be nolufe, svmplus or dp'
        assert featsel in ['nofeatsel','rfe','mi','anova','chi2','bahsic'], 'feat selection method not valid'

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
        self.stepsize = stepsize
        self.featsel = featsel
        self.classifier = classifier


        # if self.classifier == 'baseline':
        #     self.lupimethod='nolufe'
        #     self.featsel='nofeatsel'
        #     self.topk='all'
        # if self.classifier == 'featsel' or 'svmreverse':
        #     self.lupimethod='nolufe'

        if stepsize==0.1:
            self.name = '{}-{}-{}-{}selected-{}{}priv'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv)
        elif percentageofinstances != 100:
            self.name = '{}-{}-{}-{}selected-{}{}priv-{}instances'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv, self.percentageofinstances)
        else:
            self.name = '{}-{}-{}-{}selected-{}{}priv-{}'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                          self.take_top_t, self.percent_of_priv, self.stepsize)

        if self.dataset!='tech':
            self.name = '{}-{}-{}-{}selected-{}{}priv'.format(self.classifier, self.lupimethod, self.featsel, self.topk,
                                                              self.take_top_t, self.percent_of_priv)
    def print_all_settings(self):
        pprint(vars(self))
        # print(self.k,self.top)
#

# classifier = 'featselector'
# for featsel in ['bahsic']:
#     for foldnum in range(2):
#         for dataset in ['arcene','madelon','gisette','dexter','dorothea']:
#             setting = Experiment_Setting(classifier,0,'nolufe', featsel, foldnum=foldnum,dataset=dataset)
#             single_fold(setting)


# for datasetnum in range(280,295):
#     for foldnum in range(10):
#         setting = Experiment_Setting(foldnum, 300, 'tech', datasetnum, 'nolufe', 'bahsic', 1, 'featselector')
#         single_fold(setting)

#
# for datasetnum in range(40):
#     for i in range(10):
#         setting = Experiment_Setting(foldnum=i, topk=10, dataset='tech', datasetnum=datasetnum, kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=1,
#                                      percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='dp', featsel='mi', classifier='lufe')
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

# TEST

#
#
# s = Experiment_Setting(foldnum=9, topk=30, dataset='arcene', datasetnum=0, kernel='linear',
#          cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=10, take_top_t='top', lupimethod='svmplus',
#          featsel='rfe',classifier='lufe',stepsize=0.1)
#
# single_fold(s)



