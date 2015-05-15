num_folds=10


for dataset_num in range(10):
    for fold_num in range(num_folds):
        dataset='awa{}'.format(dataset_num)
        print '--k {} --num-folds {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, 10, dataset, 'linear', 0, 5,6)
