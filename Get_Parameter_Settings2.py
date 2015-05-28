num_folds=10


#
# for fold_num in range(1,11):
#     for dataset_num in range(10):
#         dataset='awa{}'.format(dataset_num)
#         print '--k {} --num-folds {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, 10, dataset, 'linear', 0, 4,5)

# for dataset in ['gisette','madelon','arcene','dorothea','dexter']:

# for awa_num in range(10):
dataset='sick'
for fold_num in range(1,11):
    for top_k_percent in [5,10,25,50,75]:
        # dataset='awa{}'.format(awa_num)
        print '--k {} --percentage {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, top_k_percent, dataset, 'linear', 0, 7,8)

