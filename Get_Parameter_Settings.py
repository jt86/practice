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
#



dataset='tech'
for top_k in [300]:#,500]:#,500]:#:,500]:#,500]:#,500]:#100,200,400,600,700,800,900,1000]:
    for take_top_t in ['top']:#,'bottom']:
        # for percentofpriv in [10,20,30,40,50,60,70,80,90]:
        for percentofpriv in [100]:
            for datasetnum in range (295): #5
                for seed in range (10):
                    for fold_num in range(10): #0
                        print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {} --percentageofinstances {} --taketopt {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', -3, 3, 7, seed, percentofpriv, 100, take_top_t))
#                         count+=1

# print(count)

# print('--k 6 --topk 300 --dataset tech --datasetnum 110 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--k 6 --topk 300 --dataset tech --datasetnum 168 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--k 3 --topk 300 --dataset tech --datasetnum 168 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 3 --percentofpriv 100 --percentageofinstances 100 --taketopt top')

# print('--k 8 --topk 300 --dataset tech --datasetnum 18 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--k 0 --topk 300 --dataset tech --datasetnum 133 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--k 7 --topk 300 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--k 0 --topk 300 --dataset tech --datasetnum 274 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top')


# dataset = 'awa'
# for top_k in [5000]:#,500]:#,500]:#100,200,400,600,700,800,900,1000]:
#     for take_top_t in ['top']:
#         for percentofpriv in [100]:#5,50,10,20,30,40,60,70,80,90]:
#             for datasetnum in [0,1,2,3,5,7,8,9]:
#                 for seed in range (10):
#                     for fold_num in range(10): #0
#                         print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {} --percentageofinstances {} --taketopt {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', -3, 3, 7, seed, percentofpriv, 100, take_top_t))
#                         count+=1

# CVPR2016_Rcode jt306$ R 500 39 3 4 --no-save < run_scripts.R
# for number_selected in [300, 500]:
#     for datasetnum in range(49):  # 5
#         for seed in range(10):
#             for fold_num in range(10):
#                 # print('R {} {} {} {} --no-save < run_scripts.R'.format(top_k,datasetnum,seed,fold_num))
#                 print('{} {} {} {}'.format(number_selected, datasetnum, seed, fold_num))
#                 count+=1
# print (count)

# print ("500 7 0 4")


# print ("--k 1 --topk 300 --dataset tech --datasetnum 96 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 161 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 172 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 172 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 172 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 172 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 173 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 173 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 173 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 173 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 0 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 6 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 174 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 176 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 6 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 178 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 0 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 3 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 179 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 180 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 180 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 180 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 180 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 0 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 0 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 6 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 181 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 0 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 8 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 182 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 3 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 3 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 6 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 7 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 6 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 183 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 9 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 6 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 8 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 9 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 4 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 0 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 5 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 184 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 0 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 6 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 2 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 3 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 4 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 5 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 2 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
# print ("--k 1 --topk 500 --dataset tech --datasetnum 185 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 3 --percentofpriv 100 --percentageofinstances 100 --taketopt top")
