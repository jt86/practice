# __author__ = 'jt306'
import numpy as np
from Get_Full_Path import get_full_path
# from scipy import sparse as sp
# from sklearn.cross_validation import StratifiedKFold
# from sklearn import preprocessing
# from sklearn import preprocessing
# # import numpy.linalg.norm
# import sys
#
#
# class0_labels = [-1]*100
# class1_labels = [1]* 100
# all_labels = np.r_[class0_labels, class1_labels]
# print (all_labels.shape)
#
# skf_seed=
# k=1
#
# skf = StratifiedKFold(all_labels, n_folds=10, shuffle=True,random_state=skf_seed)
# for fold_num, (train_index, test_index) in enumerate(skf):
#     if fold_num==k:
#         print(test_index)
#         break
labels = np.load(get_full_path('Desktop/Privileged_Data/data_Joe/labels.npy'))
data0 = np.load(get_full_path('Desktop/Privileged_Data/data_Joe/data0.npy'))
print (data0.shape)
print (data0[0])

print (labels.shape)