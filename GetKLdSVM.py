import numpy as np
import os
from Get_Full_Path import get_full_path
from scipy import stats
from sklearn import feature_selection

# def replace_zeros(array):
#     array2=array
#     print('zeros',array==0)
#     array2[array==0]=0.001
#     print(array2)
#     return(array2)



def load_dvalues(num_datasets, setting, n_top_feats, c_value, percent_of_priv, num_repeats,num_folds):
    all_f = np.zeros([num_datasets,num_repeats,num_folds])
    all_p = np.zeros([num_datasets,num_repeats,num_folds])
    for dataset_num in range(128,num_datasets):
        print (dataset_num)
        for seed_num in range (num_repeats):
            for inner_fold in range(num_folds):
                print (seed_num,inner_fold)
                f_value, p_value =get_scores_single_fold(setting, n_top_feats, c_value,
                                                                     percent_of_priv, dataset_num, seed_num, inner_fold)
                all_f[dataset_num, seed_num, inner_fold] = f_value
                all_p[dataset_num, seed_num, inner_fold] = f_value
    np.save(get_full_path('Desktop/SavedAnalysis/f_values295'),all_f)
    np.save(get_full_path('Desktop/SavedAnalysis/p_values295'), all_f)

def get_scores_single_fold(setting, n_top_feats, c_value, percent_of_priv, dataset_num,
 seed_num, inner_fold):
    d_values = np.load(get_full_path(
        'Desktop/SavedDvalues/{}-{}-{}-{}/{}-{}-{}.npy'.format(setting, n_top_feats, c_value, percent_of_priv, dataset_num,
                                                               seed_num, inner_fold)))
    labels = np.load(get_full_path('Desktop/SavedTrainLabels/{}-{}-{}.npy'.format(dataset_num, seed_num, inner_fold)))
    f_value, p_value = feature_selection.f_classif(np.reshape(d_values, [len(d_values), 1]), labels)
    return f_value, p_value


# load_dvalues(295,'dsvm',300,'cross-val',100,10,10)

def load_dvalues2(num_datasets, setting, n_top_feats, c_value, percent_of_priv, num_repeats,num_folds):
    for dataset_num in [127]:
        print(dataset_num)
        for seed_num in range(10):
            for inner_fold in range(10):
                # print(seed_num, inner_fold)
                # f_value, p_value = get_scores_single_fold(setting, n_top_feats, c_value,
                #                                           percent_of_priv, dataset_num, seed_num, inner_fold)
                try:
                    f_value, p_value = get_scores_single_fold(setting, n_top_feats, c_value,
                                                              percent_of_priv, dataset_num, seed_num, inner_fold)
                except ValueError:
                    print(seed_num,inner_fold,)
                # if not os.path.exists(output_directory):
                #     os.makedirs(output_directory)


load_dvalues2(295,'dsvm',300,'cross-val',100,10,10)