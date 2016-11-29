# import os
# path = '/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/'
# count=0
# for num_selected in [300,500]:
#     for datasetnum in range(49):
#         for seed in range(10):
#             for fold in range(10):
#                 location = 'tech{}/top{}chosen-100percentinstances/cross-validation{}/lupi-{}-{}.csv'.format(datasetnum, num_selected, seed, fold,num_selected)
#                 full_path = os.path.join(path,location)
#                 # print (full_path)
#                 if not os.path.exists(full_path):
#                     print ('print ("{} {} {} {}")'.format(num_selected, datasetnum, seed, fold))
#                     count+=1
# print (count)
#

'''
Main function used to collect results over 10x10 folds and plot two results (line and bar) comparing three settings
'''

__author__ = 'jt306'
import matplotlib
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np
import sys
from scipy import stats
import matplotlib.cm as cm
import csv

print (sys.version)
print (matplotlib.__version__)
num_repeats = 10
num_folds = 10

percent_of_priv=100
seed='linear'
method = 'RFE'
dataset='tech'

percent_of_priv = 100
percentofinstances=100
toporbottom='top'
step=0.1
lufecolor='forestgreen'
rfecolor='purple'
basecolor='dodgerblue'

###########################################################

num_datasets=295

#NB if 'method' is RFE doesn't work - delete last "-{}" from line below
experiment_name = 'dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'

np.set_printoptions(linewidth=132)

count=0
dsvm_lufe=[]
for n_top_feats in[300,500]:
    for dataset_num in range(num_datasets):
        all_folds_baseline, all_folds_SVM, all_folds_lufe1 = [], [], []
        for seed_num in range (num_repeats):
            output_directory = ('/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top{}chosen-100percentinstances/cross-validation{}'.format(dataset_num, n_top_feats,seed_num))
            for inner_fold in range(num_folds):
                if not os.path.exists(os.path.join(output_directory,'lupi-{}-{}.csv').format(inner_fold,n_top_feats)):
                    # print ('datasetnum {} seednum {} inner_fold {}'.format(dataset_num,seed_num,inner_fold))
                    # count+=1
                    # print (count)
                    print('print ("--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {} --percentageofinstances {} --taketopt {}")'.format(
                            inner_fold, n_top_feats, dataset, dataset_num, 'linear', -3, 3, 7, seed_num, percent_of_priv, 100,'top'))