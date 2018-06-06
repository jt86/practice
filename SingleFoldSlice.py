"""
This is the main function.
Things to check before running: (1) values of C, (2) output directory and whether old output is there
(3) number of jobs in go-practice-submit.sh matches desired number of settings to run in Run Experiment
(4) that there is no code test run
(5) data is regularised as desired in GetSingleFoldData
(6) params including number of folds and stepsize set correctly
"""

import os
from SVMplus import svmplusQP, svmplusQP_Predict
from ParamEstimation import get_best_params, get_best_CandCstar, get_best_params_dp2, get_best_C_Cstar_omega_omegastar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
import sys
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
import numpy as np
from New import svm_problem, svm_u_problem
from Models import SVMdp, SVMu, get_accuracy_score
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
sys.path.append('bahsic')
from bahsic import CBAHSIC,vector
# from sklearn.feature_selection import mutual_info_classif
from pprint import pprint



############## HELPER FUNCTIONS

def get_priv_subset(feature_set,take_top_t,percent_of_priv):
    """Returns either the top t% or bottom t% from an ordered feature set"""
    num_of_priv_feats = percent_of_priv * feature_set.shape[1] // 100
    assert take_top_t in ['top','bottom']
    if take_top_t == 'top':
        priv_train = feature_set[:, :num_of_priv_feats]
    if take_top_t == 'bottom':
        priv_train = feature_set[:, -num_of_priv_feats:]
    print('privileged data shape', priv_train.shape)
    return priv_train

def make_directory(directory):
    """Attempts to make a directory if it doesn't exist already"""
    try:
        os.makedirs(directory)
    except OSError:
        if not os.path.isdir(directory):
            raise
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_scores(s, score, cross_val_folder):
    """Writes the score from a classifier to a CSV, along with an index for the fold number"""
    print('{} score = {}'.format(s.classifier, score))
    with open(os.path.join(cross_val_folder, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'a') as cv_lupi_file:
        cv_lupi_file.write('{},{}'.format(s.foldnum,score) + '\n')


############## LUFe FUNCTIONS

def delta_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    """Makes calls to do param optimisation, then train and test the SVM delta+ classifier, and then save scores"""
    C, gamma, delta = get_best_params_dp2(s, normal_train, labels_train, priv_train, cross_val_folder)
    problem = svm_problem(normal_train, priv_train, labels_train, C=C,
                          gamma=gamma, delta=delta)
    s2 = SVMdp()
    dp_classifier = s2.train(prob=problem)
    dp_score = get_accuracy_score(dp_classifier, normal_test, labels_test)
    save_scores(s,dp_score,cross_val_folder)


def svm_u(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    """Makes calls to do param optimisation, then train and test the SVM U+ classifier, and then save scores"""
    C, gamma, delta = get_best_params_dp2(s, normal_train, labels_train, priv_train, cross_val_folder)
    problem = svm_u_problem(X=normal_train, Xstar=priv_train,XstarStar=priv_train, Y=labels_train, C=C,
                          gamma=gamma, delta=delta)
    s2 = SVMdp()
    dp_classifier = s2.train(prob=problem)
    dp_score = get_accuracy_score(dp_classifier, normal_test, labels_test)
    save_scores(s,dp_score,cross_val_folder)


def svm_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    """Makes calls to do param optimisation, then train and test the SVM+ classifier (linear), and then save scores"""
    c_svm_plus, c_star_svm_plus = get_best_CandCstar(s, normal_train, labels_train, priv_train, cross_val_folder)
    duals, bias = svmplusQP(normal_train, labels_train.copy(), priv_train, c_svm_plus, c_star_svm_plus, s.kernel)
    lupi_predictions = svmplusQP_Predict(normal_train, normal_test, duals, bias).flatten()
    accuracy_lupi = np.sum(labels_test == np.sign(lupi_predictions)) / (1. * len(labels_test))
    save_scores(s, accuracy_lupi, cross_val_folder)


def svm_plus_rbfcv(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    """Makes calls to do param optimisation, then train and test the SVM+ classifier (RBF), and then save scores"""
    c_svm_plus, c_star_svm_plus, omega, omegastar = get_best_C_Cstar_omega_omegastar(s, normal_train, labels_train, priv_train, cross_val_folder)
    duals, bias = svmplusQP(normal_train, labels_train.copy(), priv_train, c_svm_plus, c_star_svm_plus, s.kernel, omega, omegastar)
    lupi_predictions = svmplusQP_Predict(normal_train, normal_test, duals, bias).flatten()
    accuracy_lupi = np.sum(labels_test == np.sign(lupi_predictions)) / (1. * len(labels_test))
    save_scores(s, accuracy_lupi, cross_val_folder)

def do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    """Takes subset of privileged information as required then calls the required LUFe classiifer based on setting"""
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
    """Generates random secondary dataset, then uses in LUFe classiifer, based on setting"""
    num_instances,num_feats = priv_train.shape
    np.random.seed(s.foldnum)
    random_array = np.random.rand(num_instances, num_feats)
    print('random \n',random_array)
    random_priv_train = preprocessing.scale(random_array)
    do_lufe(s, normal_train, labels_train, random_priv_train, normal_test, labels_test, cross_val_folder)

def do_lufeshuffle(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder):
    """Shuffles the secondary dataset, then uses in LUFe classiifer, based on setting"""
    np.random.seed(s.foldnum)
    print('\n',priv_train)
    np.random.shuffle(priv_train)
    print('\n', priv_train)
    do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder)

def do_lufetrain(s, normal_train, labels_train, priv_train, cross_val_folder):
    """"Does reversed LUFe, using SVM+ with selected features as secondary dataset, unselected as primary"""
    if s.lupimethod == 'svmplus':
        svm_plus(s, normal_train, labels_train, priv_train, normal_train, labels_train, cross_val_folder)

############## FUNCTIONS TO GET SUBSETS OF FEATURES AND SUBSETS OF INSTANCES

def do_rfe(s, all_train, all_test, labels_train, labels_test,cross_val_folder):
    """Get CV'd best parameters for RFE. Fit RFE to training data and save ordered features. Call function to do RFE"""
    best_rfe_param = get_best_params(s, all_train, labels_train, cross_val_folder, 'rfe')
    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.foldnum)
    rfe = RFE(estimator=svc, n_features_to_select=s.topk, step=0.1)
    rfe.fit(all_train, labels_train)
    if s.percentageofinstances!=100:
        directory = get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-{}instances/'.format(s.topk, s.featsel,s.percentageofinstances))
    else:
        directory = get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/'.format(s.topk, s.featsel,s.percentageofinstances))
    make_directory(directory)
    np.save(get_full_path(directory+'/{}{}-{}-{}'.format(s.dataset, s.datasetnum, s.skfseed, s.foldnum)), (np.argsort(rfe.ranking_)))
    do_svm_for_rfe(s, all_train, all_test, labels_train, labels_test, cross_val_folder, best_rfe_param)


def do_svm_for_rfe(s,all_train,all_test,labels_train,labels_test,cross_val_folder, best_rfe_param):
    """Load feature subsets saved by RFE, train & deploy classifier, and save scores"""
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
    # if s.featsel=='rfe' and s.dataset=='tech' and s.percentageofinstances==100:
    #     stepsize = str(s.stepsize).replace('.', '-')
    #     ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/{}{}-{}-{}.npy'.
    #                           format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
    # elif s.featsel == 'rfe' and s.dataset == 'tech' and s.percentageofinstances!=100:
    #     stepsize = str(s.stepsize).replace('.', '-')
    #     ordered_indices = np.load(
    #         get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-step{}-{}instances/{}{}-{}-{}.npy'.
    #                       format(s.topk, s.featsel, stepsize, s.percentageofinstances,s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
    # elif s.percentageofinstances != 100:
    #     print('not 100')
    #     ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-{}instances/{}{}-{}-{}.npy'.
    #                                             format(s.topk, s.featsel, s.percentageofinstances, s.dataset, s.datasetnum, s.skfseed,
    #                                                    s.foldnum)))
    # else:

    if s.percentageofinstances==100:
        ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}/{}{}-{}-{}.npy'.
                          format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.foldnum)))
    else:
        ordered_indices = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top{}{}-{}instances/{}{}-{}-{}.npy'.
                                format(s.topk, s.featsel, s.percentageofinstances, s.dataset, s.datasetnum, s.skfseed,  s.foldnum)))

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
    print(labels_train)
    clf.fit(train_data, labels_train)
    predictions = clf.predict(test_data)
    score = accuracy_score(labels_test, predictions)
    save_scores(s, score, cross_val_folder)

def do_svm_on_train(s, all_train, all_test, labels_train, cross_val_folder):

    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)

    do_svm(s, normal_train, labels_train, normal_train, labels_train, cross_val_folder)


##################################################################################################

def single_fold(s):
    pprint(vars(s))
    np.random.seed(s.foldnum)
    output_directory = get_full_path(('Desktop/Privileged_Data/AllResults/{}/{}/{}/{}{}/')
                                     .format(s.dataset,s.kernel, s.name,s.dataset, s.datasetnum))
    print(output_directory)
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
        if s.classifier == 'lufenonlincrossval':
            svm_plus_rbfcv(s, normal_train, labels_train, priv_train, normal_test, labels_test, output_directory)
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
