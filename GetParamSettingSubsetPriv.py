num_folds=10
subset_sizes = range(5,51,5)
print subset_sizes
for dataset in ['gisette','arcene']:
    for fold_num in range(1,11):
        for subset_of_priv in subset_sizes:
            print '--k {} --subsetofpriv {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, subset_of_priv, dataset, 'linear', 0, 4,5)

