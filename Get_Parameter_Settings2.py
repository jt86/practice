num_folds=10


#
# for fold_num in range(1,11):
#     for dataset_num in range(10):
#         dataset='awa{}'.format(dataset_num)
#         print '--k {} --num-folds {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, 10, dataset, 'linear', 0, 4,5)


dataset='arcene'

for fold_num in range(1,11):
    print '--k {} --num-folds {} --dataset {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, 10, dataset, 'linear', 0, 4,5)
