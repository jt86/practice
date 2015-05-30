num_folds=10

for dataset in ['gisette','arcene']:
    for fold_num in range(1,11):
        for subset_of_priv in [10,20,30,40,50,60,70,80,90,100]:
            print '--k {} --subsetofpriv {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, subset_of_priv, dataset, 'linear', 0, 4,5)

