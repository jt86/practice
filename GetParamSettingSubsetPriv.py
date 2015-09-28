num_folds=10
subset_sizes = list(range(10,101,10))

for awanum in range(10):
    for fold_num in range(1,11):
        for subset_of_priv in subset_sizes:
            dataset='awa{}'.format(awanum)
            print('--k {} --per--subsetofpriv {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, subset_of_priv, dataset, 'linear', 0, 4,5))

