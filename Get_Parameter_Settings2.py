num_folds=10


dataset='mushroom'
for fold_num in range(7,11):
    for percentage in [5,10,25,50,75]:
        print '--k {} --percentage {} --num-folds {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, percentage, 10, dataset, 'linear', 0, 4,5)

# for dataset in ['gisette','madelon','arcene','dorothea','dexter']:

# for awa_num in range(10):
#
# for subset_of_priv in [10,20,30,40,50,60,70,80,90,100]:
# for dataset in ['dexter','dorothea']:
#     for fold_num in range(1,11):
#         for top_k_percent in [5,10,25,50,75]:
#             print '--k {} --percentage {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, top_k_percent, dataset, 'linear', 0, 4,5)
