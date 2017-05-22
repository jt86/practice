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


# print('--k 4 --topk 300 --dataset tech --datasetnum 123 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top')


seed = 1
dataset='tech'
top_k = 300
take_top_t ='top'
percentofpriv = 100

featsel = 'mi'

classifier = 'lufe'
lupimethod = 'svmplus'
for foldnum in range(10):
    for datasetnum in range (295): #5
        print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
              '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
        count+=1

classifier = 'lufereverse'
lupimethod = 'svmplus'
for foldnum in range(10):
    for datasetnum in range (295): #5
        print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
              '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
        count+=1

classifier = 'lufe'
lupimethod = 'dp'
for foldnum in range(10):
    for datasetnum in range (295): #5
        print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
              '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
        count+=1

classifier = 'lufereverse'
lupimethod = 'dp'
for foldnum in range(10):
    for datasetnum in range (295): #5
        print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
              '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
        count+=1

classifier = 'svmreverse'
lupimethod = 'nolufe'
for foldnum in range(10):
    for datasetnum in range (295): #5
        print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
              '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
        count+=1
print(count)
featsel = 'rfe'


#
# classifier = 'baseline'
# lupimethod = 'nolufe'
# featsel = 'nofeatsel'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1
#
# featsel = 'rfe'
#
# classifier = 'featselector'
# lupimethod = 'nolufe'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1
#
#
# classifier = 'svmreverse'
# lupimethod = 'nolufe'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1
#
#
#
#
# classifier = 'lufe'
# lupimethod = 'svmplus'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#
#         count+=1
#
# classifier = 'lufereverse'
# lupimethod = 'svmplus'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count += 1
#
# classifier = 'lufe'
# lupimethod = 'dp'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count += 1
#
#
# classifier = 'lufereverse'
# lupimethod = 'dp'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count += 1



# print (count)


    # setting = Experiment_Setting(foldnum=i, topk=30, dataset='tech', datasetnum=245, kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=4,
    #                              percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus', featsel='mi', classifier='baseline')

# (k, topk, dataset, datasetnum, kernel, cmin, cmax, number_of_cs, skfseed, percent_of_priv,
#                 percentageofinstances, take_top_t, lupimethod=None)



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
# print('--k 6 --topk 300 --dataset tech --datasetnum 110 --kernel linear --cmin -3 --cmax 3 --numberofcs 7 --skfseed 1 --percentofpriv 100 --percentageofinstances 100 --taketopt top')

