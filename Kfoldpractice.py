from sklearn.cross_validation import StratifiedKFold
from Heart import get_heart_data
from Vote import  get_vote_data
num_folds=5
# from MainFunctionParallelised import get_training_testing
from sklearn.cross_validation import  ShuffleSplit
from InitialFeatSelection import get_best_feats


# original_features_array,labels_array = get_heart_data()
original_features_array,labels_array = get_vote_data()


c_values = [0.1,1.,10.]
take_t=True


def get_indices_for_fold(labels_array, num_folds, fold_num):
    for index, (train,test) in enumerate(StratifiedKFold(labels_array, num_folds, shuffle=False , random_state=1)):
        if index==fold_num:
            return train,test


def get_train_test_selected_unselected(fold_num):
    train, test = get_indices_for_fold(labels_array, 5, fold_num)
    number_of_training_instances = int(len(train) - (len(train) / num_folds)) - 1
    print('number_of_training_instances', number_of_training_instances)
    print(train,test)
    all_training, all_testing = original_features_array[train], original_features_array[test]
    training_labels, testing_labels = labels_array[train], labels_array[test]
    return get_training_testing(take_t,all_training,all_testing,training_labels,c_values, num_folds,number_of_training_instances)



def get_training_testing(take_t,all_training,all_testing,training_labels,c_values, num_folds,number_of_training_instances):
    if take_t == True:
        print('taking top t only')
        rs = ShuffleSplit((number_of_training_instances - 1), n_iter=10, test_size=.2, random_state=0)
        top_t_indices, remaining_indices = get_best_feats(all_training,training_labels,c_values, num_folds, rs, 'heart')
        top_t_training, unselected_features_training = all_training[:,top_t_indices], all_training[:,remaining_indices]
        top_t_testing, unselected_features_testing = all_testing[:,top_t_indices], all_testing[:,remaining_indices]

    ######################
    else:
        top_t_training = all_training
        top_t_testing = all_testing
        unselected_features_training, unselected_features_testing = None,None

    return top_t_training,top_t_testing, unselected_features_training, unselected_features_testing



print(get_train_test_selected_unselected(4))