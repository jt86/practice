#hello hello
dataset='tech'

count=1
for seed in range (10):  #4
    for top_k in [300]:#,500]:#100,200,400,600,700,800,900,1000]:
        for datasetnum in range (49): #5
            for fold_num in range(10): #0
                print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', -3, 3, 7, seed, 100))

                count+=1

# print('--k 0 --topk 300 --dataset "tech" --datasetnum 5  --kernel "linear" --cmin 0 --cmax 4 --numberofcs 5 --skfseed 4')


# for dataset in ['madelon','arcene','dorothea','dexter','gisette']:
#     for fold_num in range(1,11):
#         for top_k_percent in [5,10,25,50,75]:
#             print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {}'.format(fold_num, top_k_percent, dataset, 0, 'linear', 0, 4,5))
#







