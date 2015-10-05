
dataset='tech'


for datasetnum in range (49):
    for fold_num in range(1,11):
        for top_k in [100,200,300,400,500,600,700,800,900,1000]:
            print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', 0, 4,5))

#
# dataset = 'dexter'
# datasetnum=0
# for fold_num in range(1,11):
#     for top_k_percent in [5,10,25,50,75]:
#         print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, top_k_percent, dataset, datasetnum, 'linear', 0, 4,5))




# for dataset in ['gisette','madelon','arcene','dorothea','dexter']:

# for awa_num in range(10):
#
# for subset_of_priv in [10,20,30,40,50,60,70,80,90,100]:
# for dataset in ['dexter','dorothea']:
