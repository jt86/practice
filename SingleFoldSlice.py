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
from ParamEstimation import get_best_C, get_best_RFE_C, get_best_CandCstar, get_best_params_dp#, get_best_params_dp2
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
from sklearn.feature_selection import SelectPercentile, f_classif, chi2#, mutual_info_classif

# print (PYTHONPATH)

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

def delta_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder2, classifier):
    C, gamma, delta = get_best_params_dp(s, normal_train, labels_train, priv_train, cross_val_folder2)
    problem = svm_problem(normal_train, priv_train, labels_train, C=C,
                          gamma=gamma, delta=delta)
    s2 = SVMdp()
    dp_classifier = s2.train(prob=problem)
    dp_score = get_accuracy_score(dp_classifier, normal_test, labels_test)
    with open(os.path.join(cross_val_folder2, '{}-{}-{}.csv'.format(classifier, s.k, s.topk)), 'a') as cv_lupi_file:
        cv_lupi_file.write(str(dp_score) + ',')



def svm_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder2, classifier):
    c_svm_plus, c_star_svm_plus = get_best_CandCstar(s, normal_train, labels_train, priv_train, cross_val_folder2)
    duals, bias = svmplusQP(normal_train, labels_train.copy(), priv_train, c_svm_plus, c_star_svm_plus)
    lupi_predictions = svmplusQP_Predict(normal_train, normal_test, duals, bias).flatten()
    accuracy_lupi = np.sum(labels_test == np.sign(lupi_predictions)) / (1. * len(labels_test))
    with open(os.path.join(cross_val_folder2, '{}-{}-{}.csv'.format(classifier, s.k, s.topk)), 'a') as cv_lupi_file:
        cv_lupi_file.write(str(accuracy_lupi) + ',')



def do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder, classifier):
    cross_val_folder2 = os.path.join(cross_val_folder, '{}-{}'.format(s.take_top_t, s.percent_of_priv))
    make_directory(cross_val_folder2)
    priv_train = get_priv_subset(priv_train, s.take_top_t, s.percent_of_priv)

    if s.lupimethod == 'dp':
        delta_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder2, classifier)
    if s.lupimethod == 'svmplus':
        svm_plus(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder2, classifier)

def do_rfe(s, all_train, all_test, labels_train, labels_test,cross_val_folder):
    best_rfe_param = get_best_RFE_C(s, all_train, labels_train)
    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.k)
    rfe = RFE(estimator=svc, n_features_to_select=s.topk, step=s.stepsize)
    rfe.fit(all_train, labels_train)
    best_n_mask = rfe.support_
    np.save(get_full_path('Desktop/Privileged_Data/SavedIndices/top{}{}/{}{}-{}-{}'.
                          format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.k)), best_n_mask)
    normal_train = all_train[:, best_n_mask].copy()
    normal_test = all_test[:, best_n_mask].copy()
    priv_train = all_train[:, np.invert(best_n_mask)].copy()
    priv_test = all_test[:, np.invert(best_n_mask)].copy()
    # all_features_ranking = rfe.ranking_[np.invert(best_n_mask)] # gets just the unselected features' rankings
    # priv_train = priv_train[:, np.argsort(all_features_ranking)] #reorder train and test using this
    # priv_test = priv_test[:, np.argsort(all_features_ranking)]
    return normal_train, normal_test, priv_train, priv_test


def do_mutinfo(s, all_train, labels_train, all_test):
    scores = mutual_info_classif(all_train, labels_train)
    ordered_indices = np.array(np.argsort(scores)[::-1])  # ordered feats is np array of indices from biggest to smallest

    sorted_training = all_train[:, ordered_indices]
    sorted_testing = all_test[:, ordered_indices]

    normal_train = sorted_training[:, :s.topk]  # take the first n_top_feats
    normal_test = sorted_testing[:, :s.topk]
    priv_train = sorted_training[:, s.topk:]
    priv_test = sorted_testing[:, s.topk:]

    np.save(get_full_path('Desktop/Privileged_Data/SavedIndices/top{}{}/{}{}-{}-{}'.
                          format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.k)), ordered_indices[:s.topk])

    return normal_train, normal_test, priv_train, priv_test


def do_svm_for_rfe(s, normal_train, normal_test, labels_train, labels_test, cross_val_folder, best_rfe_param, best_n_mask):

    svc = SVC(C=best_rfe_param, kernel=s.kernel, random_state=s.k)
    svc.fit(normal_train, labels_train)
    rfe_accuracy = svc.score(normal_test, labels_test)

    with open(os.path.join(cross_val_folder, 'svm-{}-{}.csv'.format(s.k, s.topk)), 'a') as cv_svm_file:
        cv_svm_file.write(str(rfe_accuracy) + ",")

def do_featsel(s, all_train, all_test, labels_train, labels_test,cross_val_folder,featsel):
    if featsel == 'RFE':
        do_rfe(s, all_train, all_test, labels_train, labels_test,cross_val_folder)
    if featsel == 'MI':
        do_mutinfo(all_train, labels_train)


def do_baseline(s, all_train, labels_train, all_test, labels_test, cross_val_folder):
    best_C_baseline = get_best_C(s, all_train, labels_train, cross_val_folder)
    print('all training shape', all_train.shape)
    clf = svm.SVC(C=best_C_baseline, kernel=s.kernel, random_state=s.k)
    clf.fit(all_train, labels_train)
    baseline_predictions = clf.predict(all_test)
    print('baseline', accuracy_score(labels_test, baseline_predictions))
    with open(os.path.join(cross_val_folder, 'baseline-{}.csv'.format(s.k)), 'a') as baseline_file:
        baseline_file.write(str(accuracy_score(labels_test, baseline_predictions)) + ',')

def load_saved(s):
    best_n_mask = np.load(get_full_path('Desktop/Privileged_Data/SavedIndices/top{}{}/{}{}-{}-{}'.
                                        format(s.topk, s.featsel, s.dataset, s.datasetnum, s.skfseed, s.k)))
    return best_n_mask



def do_unselected_svm(s, priv_train, priv_test, labels_train, labels_test, cross_val_folder):
    # best_n_mask = load_saved(s)
    # unselected_train = all_train[:, np.invert(best_n_mask)]
    # unselected_test = all_test[:, np.invert(best_n_mask)]
    best_C = get_best_C(s, priv_train, labels_train, cross_val_folder)
    clf = svm.SVC(C=best_C, kernel=s.kernel, random_state=s.k)
    clf.fit(priv_train, labels_train)
    baseline_predictions = clf.predict(priv_test)
    print('baseline', accuracy_score(labels_test, baseline_predictions))
    with open(os.path.join(cross_val_folder, 'unselectedsvm-{}.csv'.format(s.k)), 'a') as baseline_file:
        baseline_file.write(str(accuracy_score(labels_test, baseline_predictions)) + ',')


def do_svm_for_mutinfo(s, normal_train, labels_train, normal_test, labels_test, cross_val_folder):
    best_param = get_best_C(s, normal_train, labels_train, cross_val_folder)
    svc = SVC(C=best_param, kernel=s.kernel, random_state=s.k)
    svc.fit(normal_train, labels_train)
    svm_accuracy = svc.score(normal_test, labels_test)
    with open(os.path.join(cross_val_folder, 'svm-{}-{}.csv'.format(s.k, s.topk)), 'a') as cv_svm_file:
        cv_svm_file.write(str(svm_accuracy) + ",")

##################################################################################################

def single_fold(s):
    print(s.cvalues)
    print('{}% of train instances; {}% of discarded feats used as priv'.format(s.percentageofinstances,s.percent_of_priv))
    np.random.seed(s.k)
    output_directory = get_full_path((
                                     'Desktop/Privileged_Data/LUFe-{}-{}-{}-{}percentinstances/{}{}/top{}chosen-{}percentinstances').format(
        s.dataset, s.lupimethod, s.featsel, s.percentageofinstances, s.dataset, s.datasetnum, s.topk, s.percentageofinstances))

    make_directory(output_directory)
    param_estimation_file = open(os.path.join(output_directory, 'param_selection.csv'), "a")
    cross_val_folder = os.path.join(output_directory, 'cross-validation{}'.format(s.skfseed))
    make_directory(cross_val_folder)

    all_train, all_test, labels_train, labels_test = get_train_and_test_this_fold(s)
    param_estimation_file.write("\n\n n={},fold={}".format(s.topk, s.k))


    # RFE returns which feature are normal, and which are priv

    do_baseline(s, all_train, labels_train, all_test, labels_test, cross_val_folder)

    if s.lupimethod == 'svmplus':

        if s.featsel == 'RFE':
            normal_train, normal_test, priv_train, priv_test = do_rfe(s,all_train,all_test,labels_train,labels_test,cross_val_folder)

            do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder, 'lufe')
            do_lufe(s, priv_train, labels_train, normal_train, priv_test, labels_test, cross_val_folder, 'reversedlufe')
            do_unselected_svm(s, priv_train, priv_test, labels_train, labels_test, cross_val_folder)

        if s.featsel == 'MI':
            normal_train, normal_test, priv_train, priv_test = do_mutinfo(all_train, labels_train)

            do_svm_for_mutinfo(s, normal_train, labels_train, normal_test, labels_test, cross_val_folder)
            do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder, 'lufe')
            do_lufe(s, priv_train, labels_train, normal_train, priv_test, labels_test, cross_val_folder, 'reversedlufe')
            do_unselected_svm(s, priv_train, priv_test, labels_train, labels_test, cross_val_folder)

    if s.lupimethod == 'dp':
        if s.featsel == 'RFE':
            normal_train, normal_test, priv_train, priv_test = do_rfe(s,all_train,all_test,labels_train,labels_test,cross_val_folder)

            do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder, 'lufe')
            do_lufe(s, priv_train, labels_train, normal_train, priv_test, labels_test, cross_val_folder, 'reversedlufe')
            do_unselected_svm(s, priv_train, priv_test, labels_train, labels_test, cross_val_folder)

        if s.featsel == 'MI':
            normal_train, normal_test, priv_train, priv_test = do_mutinfo(s, all_train, labels_train, all_test)

            do_svm_for_mutinfo(s, normal_train, labels_train, normal_test, labels_test, cross_val_folder)
            do_lufe(s, normal_train, labels_train, priv_train, normal_test, labels_test, cross_val_folder, 'lufe')
            do_lufe(s, priv_train, labels_train, normal_train, priv_test, labels_test, cross_val_folder, 'reversedlufe')
            do_unselected_svm(s, priv_train, priv_test, labels_train, labels_test, cross_val_folder)

##################################################################################################

def get_random_array(num_instances, num_feats):
    random_array = np.random.rand(num_instances, num_feats)
    random_array = preprocessing.scale(random_array)
    return random_array



####### This part takes a subset of training instances

def take_subset(all_training, training_labels, percentageofinstances):
    orig_num_train_instances = all_training.shape[0]
    num_of_train_instances = orig_num_train_instances * percentageofinstances // 100
    indices = np.random.choice(orig_num_train_instances, num_of_train_instances, replace=False)
    all_training = all_training.copy()[indices, :]
    training_labels = training_labels[indices]
    print(all_training.shape)
    print(training_labels.shape)
    print(indices)


class Experiment_Setting:
    def __init__(self, k, topk, dataset, datasetnum, kernel, cvalues, skfseed,
            percent_of_priv, percentageofinstances, take_top_t, lupimethod, featsel):
        self.k = k
        self.topk = topk
        self.dataset = dataset
        self.datasetnum = datasetnum
        self.kernel = kernel
        self.cvalues = np.logspace(*[int(item)for item in cvalues.split('a')])
        # self.cvalues = (np.logspace(*cvalues))
        self.skfseed = skfseed
        self.percent_of_priv = percent_of_priv
        self.percentageofinstances = percentageofinstances
        self.take_top_t = take_top_t
        self.lupimethod =lupimethod
        self.stepsize=0.1
        self.featsel = featsel


# setting = Experiment_Setting(k=3, topk=300, dataset='tech', datasetnum=245, kernel='linear', cvalues='-3,3,1', skfseed=4,
#             percent_of_priv=100, percentageofinstances=100, take_top_t='bottom', lupimethod='dp', featsel='RFE')
# single_fold(setting)

data = (np.load('/Volumes/LocalDataHD/j/jt/jt306/Desktop/SavedIndices/top300RFE/tech0-0-0.npy'))
cvalues = '-3a3a7'
print([int(item)for item in cvalues.split('a')])
seed = 1
dataset='tech'
top_k = 300
take_top_t ='top'
percentofpriv = 100

for fold_num in range(10):
    for lupimethod in ['dp','svmplus']:
        for featsel in ['MI']:
            for datasetnum in range (295): #5
                setting = Experiment_Setting(k=fold_num, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                         cvalues=cvalues, skfseed=0, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod=lupimethod,
                         featsel=featsel)
                single_fold(setting)
