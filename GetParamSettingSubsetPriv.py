num_folds=10
subset_sizes = range(10,101,10)

for dataset in ['mushroom','gisette','arcene','dexter']:
    for fold_num in range(1,11):
        for subset_of_priv in subset_sizes:
            print '--k {} --subsetofpriv {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, subset_of_priv, dataset, 'linear', 0, 4,5)

