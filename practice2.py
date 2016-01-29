import os
import numpy as np
from sklearn.metrics import accuracy_score
from SVMplus4 import svmplusQP, svmplusQP_Predict
from ParamEstimation import  get_best_C, get_best_RFE_C, get_best_CandCstar
from sklearn import svm
from Get_Full_Path import get_full_path
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from GetSingleFoldData import get_train_and_test_this_fold
# from GetFeatSelectionData import get_train_and_test_this_fold
import sys
import numpy.random as random
from sklearn import preprocessing
np.set_printoptions(linewidth=500)
best_rfe_param= 1



class0_labels = [-1]*10
class1_labels = [1]* 10
training_labels = np.r_[class0_labels, class1_labels]

print (training_labels.shape)

ones = np.ones((10,5))
zeros = np.zeros((10,5))
useful_feats = np.concatenate((ones,zeros),axis=0)

# semi_useful_feat = np.concatenate((np.ones((11,1)),np.zeros((9,1))),axis=0)
# print('semi shape',semi_useful_feat.shape)
semi_random = np.concatenate((np.ones((10,1)),np.zeros((10,1))),axis=0)
np.concatenate((np.ones((11,1)),np.zeros((9,1))),axis=0)
noise = random.rand(20,1)/100
print ('noise',noise)

semi_useful_feat=semi_random+noise


random_feats = random.rand(20,6)


all_training  = np.concatenate((useful_feats,semi_useful_feat,random_feats),axis=1)
print (all_training)
print(all_training.shape)

#
svc = SVC(C=best_rfe_param, kernel='linear', random_state=1)
rfe = RFE(estimator=svc, n_features_to_select=5, step=1)
print ('rfe step size',rfe.step)
rfe.fit(all_training, training_labels)
# print (all_testing.shape,testing_labels.shape)
print ('num of chosen feats',sum(x == 1 for x in rfe.support_))
#
privileged_features_training=all_training[:,np.invert(rfe.support_)].copy()
best_n_mask = rfe.support_
print('best n mask',best_n_mask)



print ('rfe ranking',rfe.ranking_)


all_feats_ranked = all_training[:,np.argsort(rfe.ranking_)]
print (all_feats_ranked)


all_features_ranking = rfe.ranking_[np.invert(best_n_mask)]
privileged_features_training = privileged_features_training[:,np.argsort(all_features_ranking)]
num_of_priv_feats=1
take_top_t = 'bottom'
if take_top_t=='top':
        privileged_features_training = privileged_features_training[:,:num_of_priv_feats]
if take_top_t=='bottom':
        privileged_features_training = privileged_features_training[:,-num_of_priv_feats:]

print ('privileged data shape',privileged_features_training.shape)
print ('privileged data',privileged_features_training)

# privileged_features_training = privileged_features_training[:,np.argsort(all_features_ranking)]
# print (all_features_ranking)

