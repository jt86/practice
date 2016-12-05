'''
Generate list of parameters. Called by RunExperiment.py when running on cluster
'''

# from SaveDataForR import save_instance_and_feature_indices_for_R, save_dataset_for_R
count=0

# dataset='tech'
# for top_k in [300,500]:#,500]:#,500]:#100,200,400,600,700,800,900,1000]:
#     for take_top_t in ['top']:#,'bottom']:
#         # for percentofpriv in [10,20,30,40,50,60,70,80,90]:
#         for percentofpriv in [100]:
#             for datasetnum in range (49,92):
#                 # save_dataset_for_R(datasetnum)
#                 for seed in range (10):
#                     for fold_num in range(10): #0
#                         print('--k {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin {} --cmax {} --numberofcs {} --skfseed {} --percentofpriv {} --percentageofinstances {} --taketopt {}'.format(fold_num, top_k, dataset, datasetnum, 'linear', -3, 3, 7, seed, percentofpriv, 100, take_top_t))
#                         count+=1
# print(count)
#                         save_instance_and_feature_indices_for_R(k=fold_num, dataset=dataset, topk=top_k,
#                                                                 datasetnum=datasetnum,kernel='linear', cmin=-3, cmax=3,
#                                                                 number_of_cs=7, skfseed=seed)




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
for number_selected in [300, 500]:
    for datasetnum in range(49,295):  # 5
        for seed in range(10):
            for fold_num in range(10):
                # print('R {} {} {} {} --no-save < run_scripts.R'.format(top_k,datasetnum,seed,fold_num))
                print('{} {} {} {}'.format(number_selected, datasetnum, seed, fold_num))
                count+=1
print (count)

# print ("500 7 0 4")
# print ("500 7 0 9")
# print ("500 7 1 6")
# print ("500 7 2 5")
# print ("500 7 2 8")
# print ("500 7 3 7")
# print ("500 7 3 9")
# print ("500 7 4 7")
# print ("500 7 5 4")
# print ("500 7 5 9")
# print ("500 7 6 4")
# print ("500 7 7 6")
# print ("500 7 7 9")
# print ("500 7 8 9")
# print ("500 7 9 4")
# print ("500 8 1 0")
# print ("500 8 4 0")
# print ("500 8 7 0")
# print ("500 8 8 0")
#
# print ("500 6 0 6")
# print ("500 6 1 7")
# print ("500 6 2 3")
# print ("500 6 2 5")
# print ("500 6 3 6")
# print ("500 6 3 7")
# print ("500 6 3 8")
# print ("500 6 3 9")
# print ("500 6 4 8")
# print ("500 6 5 8")
# print ("500 6 6 6")
# print ("500 6 7 5")
# print ("500 6 8 6")
# print ("500 6 9 2")
# print ("500 6 9 5")
# print ("500 6 9 9")
# print ("500 6 0 5")
