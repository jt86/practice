from Get_Full_Path import get_full_path
import numpy as np
from GetSingleFoldData import get_train_and_test_this_fold

def single_fold(k, topk, dataset, datasetnum, kernel, cmin, cmax, number_of_cs, skfseed, percent_of_priv,
                percentageofinstances, take_top_t, lupimethod=None):
    indices = np.load(get_full_path('Desktop/Privileged_Data/SavedIndices/top{}RFE/{}{}-{}-{}.npy'.format(topk,dataset,datasetnum,skfseed,k)))
    all_training, all_testing, training_labels, testing_labels = get_train_and_test_this_fold(dataset, datasetnum, k,
                                                                                              skfseed)
    normal_features_testing = all_testing[:, indices].copy()
    privileged_features_training = all_training[:, np.invert(indices)].copy()


    print(normal_features_testing)
