#hello hello
dataset='tech'

for seed in range (10):  #4
    for top_k in [500]:#,500]:#100,200,400,600,700,800,900,1000]:
        for datasetnum in range (49): #5
            for fold_num in range(4): #0
                print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', 0, 4,5, seed, 100))


# print('--k 0 --topk 300 --dataset "tech" --datasetnum 5  --kernel "linear" --cmin 0 --cmax 4 --numberofcs 5 --skfseed 4')





#
# for dataset in ['madelon','arcene','dorothea','dexter','gisette']:
#     for fold_num in range(1,11):
#         for top_k_percent in [5,10,25,50,75]:
#             print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, top_k_percent, dataset, 0, 'linear', 0, 4,5))
#




# for awa_num in range(10):

# for subset_of_priv in [10,20,30,40,50,60,70,80,90,100]:
# for dataset in ['dexter','dorothea']:
