__author__ = 'jt306'

from SingleFold2 import single_fold

num_folds=10
for fold_num in range(num_folds):
    # for dataset_num in range(10):
    #     dataset='awa{}'.format(dataset_num)
    print(fold_num)
    single_fold(k=fold_num, num_folds=num_folds,  dataset='awa0', peeking=False, kernel='linear', cmin=0, cmax=5, number_of_cs=6)
