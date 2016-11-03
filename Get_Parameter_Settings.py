'''
Generate list of parameters. Called by RunExperiment.py when running on cluster
'''


count=1
# for dataset in ['arcene','dexter','gisette','dorothea','madelon']:
#     for seed in range(10):
#         for fold_num in range(10):
#             for top_k_percent in [5,10,25,50,75]:
#                 print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {}'.format(fold_num, top_k_percent, dataset, 0, 'linear', 0, 3, 4, seed, 100))
#                 count+=1
#

dataset='tech'
for top_k in [300,500]:#,500]:#,500]:#100,200,400,600,700,800,900,1000]:
    for take_top_t in ['top']:#,'bottom']:
        # for percentofpriv in [10,20,30,40,50,60,70,80,90]:
        for percentofpriv in [100]:
            for datasetnum in range (49): #5
                for seed in range (10):
                    for fold_num in range(10): #0
                        print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {} --percentageofinstances {} --taketopt {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', -3, 3, 7, seed, percentofpriv, 100, take_top_t))
                        count+=1
print(count)

# dataset = 'awa'
# for top_k in [5000]:#,500]:#,500]:#100,200,400,600,700,800,900,1000]:
#     for take_top_t in ['top']:
#         for percentofpriv in [100]:#5,50,10,20,30,40,60,70,80,90]:
#             for datasetnum in [0,1,2,3,5,7,8,9]:
#                 for seed in range (10):
#                     for fold_num in range(10): #0
#                         print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {} --percentageofinstances {} --taketopt {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', -3, 3, 7, seed, percentofpriv, 100, take_top_t))
#                         count+=1

# #CVPR2016_Rcode jt306$ R 500 39 3 4 --no-save < run_scripts.R
# for top_k in [300,500]:
#     for datasetnum in range(49):  # 5
#         for seed in range(1):
#             for fold_num in range(1):
#                 print('R {} {} {} {} --no-save < run_scripts.R'.format(top_k,datasetnum,seed,fold_num))