import numpy as np
import os
from Get_Full_Path import get_full_path
from scipy import stats
from sklearn import feature_selection
from ProcessNumpyResults import compare_two_settings_ind_folds, Setting
import matplotlib.pyplot as plt

# def replace_zeros(array):
#     array2=array
#     print('zeros',array==0)
#     array2[array==0]=0.001
#     print(array2)
#     return(array2)
svm_baseline = Setting(295,'svm',300,'cross-val',100)
dsvm_crossval = Setting(295,'dsvm',300,'cross-val',100)
improvements_list = compare_two_settings_ind_folds(svm_baseline,dsvm_crossval)
print(len(improvements_list))

def save_f_values(setting, num_repeats,num_folds):
    all_f = np.zeros([setting.num_datasets,num_repeats,num_folds])
    all_p = np.zeros([setting.num_datasets,num_repeats,num_folds])
    all_nums = np.zeros([setting.num_datasets,num_repeats,num_folds])
    for dataset_num in range(setting.num_datasets):
        print (dataset_num)
        for seed_num in range (num_repeats):
            for inner_fold in range(num_folds):
                # print (seed_num,inner_fold)
                f_value, p_value,num_of_big_dvalues =get_scores_single_fold(setting, dataset_num, seed_num, inner_fold)
                all_f[dataset_num, seed_num, inner_fold] = f_value
                all_p[dataset_num, seed_num, inner_fold] = p_value
                all_nums[dataset_num, seed_num, inner_fold]=num_of_big_dvalues
    # return(all_nums)
    # np.save(get_full_path('Desktop/SavedAnalysis/f_values295'),all_f)
    # np.save(get_full_path('Desktop/SavedAnalysis/p_values295'), all_p)
    # np.save(get_full_path('Desktop/SavedAnalysis/f_values295'), all_f)
    np.save(get_full_path('Desktop/SavedAnalysis/dvaluesunder10'), all_nums)


def get_scores_single_fold(setting, dataset_num,
 seed_num, inner_fold):
    d_values = np.load(get_full_path(
        'Desktop/SavedDvalues/{}-{}-{}-{}/{}-{}-{}.npy'.format(setting.classifier_type, setting.n_top_feats, setting.c_value, setting.percent_of_priv, dataset_num,
                                                               seed_num, inner_fold)))
    num_of_big_dvalues = (len(np.where(abs(d_values)<10)[0]))
    # print(d_values)
    #  print (len(d_values),np.count_nonzero(d_values),len(d_values)-np.count_nonzero(d_values))
    labels = np.load(get_full_path('Desktop/SavedTrainLabels/{}-{}-{}.npy'.format(dataset_num, seed_num, inner_fold)))
    f_value, p_value = feature_selection.f_classif(np.reshape(d_values, [len(d_values), 1]), labels)
    return f_value, p_value, num_of_big_dvalues

dsvm_crossval = Setting(295,'dsvm',300,'cross-val',100)
save_f_values(dsvm_crossval,10,10)


all_nums= np.load(get_full_path('Desktop/SavedAnalysis/dvaluesunder10.npy'))
all_f_values = np.load(get_full_path('Desktop/SavedAnalysis/f_values295.npy')).reshape(295,100)
all_p_values = np.load(get_full_path('Desktop/SavedAnalysis/p_values295.npy')).reshape(295,100)

print(all_nums.shape)
print(improvements_list.shape)
plt.scatter(all_nums,improvements_list)
plt.xlabel('num of deviation values < 10')
plt.ylabel('performance relative to SVM baseline')
plt.savefig(get_full_path('Desktop/SavedAnalysis/dev_values_vs_performance_10'))
plt.show()
# print(all_f_values.shape)
# print(all_p_values.shape)
# fig = plt.figure()
# ax = fig.add_subplot(2,1,1)
# ax.set_xscale('log')
#
# ax.scatter(all_f_values,improvements_list)
# # plt.show()
