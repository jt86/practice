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


# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 3 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 4 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 7 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 3 --skfseed 1 --lupimethod nolufe --featsel anova --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 4 --skfseed 1 --lupimethod nolufe --featsel anova --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 7 --skfseed 1 --lupimethod nolufe --featsel anova --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 3 --skfseed 1 --lupimethod nolufe --featsel chi2 --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 4 --skfseed 1 --lupimethod nolufe --featsel chi2 --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')
# print('--foldnum 2 --topk 300 --dataset tech --datasetnum 7 --skfseed 1 --lupimethod nolufe --featsel chi2 --classifier featselector --stepsize 0.1 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 10 --taketopt top')


seed = 1
dataset='tech'
# top_k = 300
datasetnum=0
#
# classifier = 'lufe'
# for lupimethod in  ['svmplus','dp']:
#     for featsel in ['rfe','anova','chi2']:
#         for dataset in ['arcene', 'madelon', 'gisette', 'dexter']:
#                 for foldnum in range(10):
#                     print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                         .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#                     count+=1
# print(count)
#

#
# classifier = 'featselector'
# for featsel in ['rfe','anova','chi2','bahsic']:#,'mi']:#
#     for datasetnum in range(10):
#         for foldnum in range(10):
#             for instances in [10,20,30,40,50,60,70,80,90]:
#                 print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top'
#                     .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier, 0.1, instances))
#                 count+=1
#     print(count)
#
# classifier = 'svmtrain'
# for featsel in ['rfe']:#,'anova','chi2','bahsic']:#,'mi']:#
#     for datasetnum in [5,6,8]:
#         for foldnum in range(10):
#             for instances in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
#                 print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top'
#                     .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier, 0.1, instances))
#                 count+=1
#     print(count)
#
# classifier = 'lufetrain'
# for featsel in ['rfe']:#,'anova','chi2','bahsic']:#,'mi']:#
#     for datasetnum in [5,6,8]:
#         for foldnum in range(10):
#             for instances in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
#                 print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top'
#                     .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier, 0.1, instances))
#                 count+=1
#     print(count)


# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 1 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 2 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 13 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 16 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 21 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 27 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 35 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 38 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 8 --topk 300 --dataset tech --datasetnum 53 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 54 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 55 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 66 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 73 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 7 --topk 300 --dataset tech --datasetnum 74 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 6 --topk 300 --dataset tech --datasetnum 81 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 84 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 7 --topk 300 --dataset tech --datasetnum 100 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 101 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 7 --topk 300 --dataset tech --datasetnum 152 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 5 --topk 300 --dataset tech --datasetnum 160 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 8 --topk 300 --dataset tech --datasetnum 162 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 163 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 165 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 8 --topk 300 --dataset tech --datasetnum 166 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 178 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 6 --topk 300 --dataset tech --datasetnum 184 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 187 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 7 --topk 300 --dataset tech --datasetnum 191 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 205 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 219 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 4 --topk 300 --dataset tech --datasetnum 221 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 1 --topk 300 --dataset tech --datasetnum 236 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 260 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 6 --topk 300 --dataset tech --datasetnum 261 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 262 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 9 --topk 300 --dataset tech --datasetnum 268 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 269 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 6 --topk 300 --dataset tech --datasetnum 270 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 271 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 7 --topk 300 --dataset tech --datasetnum 274 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 276 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 7 --topk 300 --dataset tech --datasetnum 277 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 8 --topk 300 --dataset tech --datasetnum 278 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 279 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 280 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 284 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 289 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 290 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 291 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')
# print('--foldnum 0 --topk 300 --dataset tech --datasetnum 294 --skfseed 1 --lupimethod nolufe --featsel rfe --classifier featselector --stepsize 0.001 --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top')

#
# classifier = 'lufe'
# featsel = 'bahsic'
# for lupimethod in ['svmplus','dp','dsvm']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print(
#                 '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#             count+=1

# classifier = 'lufereverse'
# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for lupimethod in ['svmplus','dsvm','dp']:
#         for foldnum in range(10):
#             for datasetnum in range(295):  # 5
#                 print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                     .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#                 count+=1
# print(count)
#
# classifier = 'baseline'
# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier))
#             count+=1
# print(count)
#
# classifier = 'featselector'
# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier))
#             count+=1
# print(count)
#
# classifier = 'lufe'
# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier))
#             count+=1


################# FEATSEL CHALLENGE DATASETS

datasetnum=0


for dataset in ['arcene','madelon','dexter','dorothea','gisette']:
    for foldnum in range(10):
        for top_k in range(10,100,10):
            print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
                .format(foldnum, top_k, dataset, datasetnum, seed,'nolufe', 'rfe', 'featselector'))
            count+=1

for dataset in ['arcene','madelon','dexter','dorothea','gisette']:
    for foldnum in range(10):
        for top_k in range(10,100,10):
            print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
                .format(foldnum, top_k, dataset, datasetnum, seed,'svmplus', 'rfe', 'lufe'))
            count+=1

print(count)
# classifier = 'featselector'
# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for foldnum in range(10):
#         for dataset in ['arcene','madelon','gisette','dexter','dorothea']:  # 5
#             print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, 'nolufe', featsel, classifier))
#             count+=1
#
#
# classifier = 'lufe'
# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     for lupimethod in  ['svmplus','dp']:
#         for dataset in ['arcene', 'madelon', 'gisette', 'dexter', 'dorothea']:
#                 for foldnum in range(10):
#                     print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                         .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#                     count+=1
#
# print(count)



# classifier = 'lufe'
# for featsel in ['bahsic']:#,'mi']:#'anova', 'chi2', 'mi']:
#     for percentofpriv in [10,25,50,75]:
#         for lupimethod in ['svmplus']:
#             for foldnum in range(10):
#                 for datasetnum in range(295):  # 5
#                     print(
#                         '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv {} --percentageofinstances 100 --taketopt bottom'
#                             .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier, percentofpriv))
#                     count+=1
# print(count)
#######################################
        #
# classifier = 'featselector'
# lupimethod = 'nolufe'
# for featsel in ['rfe','anova', 'chi2']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print(
#                 '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#             count += 1
#
#
# classifier = 'lufe'
# for featsel in ['mi']:
#     for lupimethod in  ['svmplus','dp']:
#         for foldnum in range(10):
#             for datasetnum in range(295):  # 5
#                 print(
#                     '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                     .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#                 count+=1
# print (count)

# classifier = 'lufe'
# lupimethod = 'dp'
# for featsel in ['rfe','anova', 'chi2','mi']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print(
#                 '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#             count += 1

# classifier = 'featselector'
# lupimethod = 'nolufe'
# featsel = 'bahsic'
# for foldnum in range(10):
#     for datasetnum in range(295):  # 5
#         print(
#             '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#             .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#         count += 1
#
#
# classifier = 'lufe'
# featsel = 'bahsic'
# for lupimethod in ['svmplus','dp']:
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print(
#                 '--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances 100 --taketopt top'
#                 .format(foldnum, top_k, dataset, datasetnum, seed, lupimethod, featsel, classifier))
#             count += 1
#     print(count)

            # featsel = 'bahsic'
# for classifier in ['featselector', 'svmreverse']:
#     lupimethod = 'nolufe'
#     for foldnum in range(10):
#         for datasetnum in range(295):  # 5
#             print(
#                 '--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                 '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset,datasetnum, 'linear', seed,percentofpriv, 100, take_top_t,lupimethod, featsel,classifier))
#             count += 1
#
#
# for classifier in ['lufe', 'lufereverse']:
#     for lupimethod in ['svmplus', 'dp']:
#         for foldnum in range(10):
#             for datasetnum in range(295):  # 5
#                 print(
#                     '--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                     '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset,datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod,featsel, classifier))
#                 count += 1
#
# print(count)




#
# for classifier in ['featselector','svmreverse']:
#     for featsel in ['anova', 'chi2']:
#         lupimethod = 'nolufe'
#         for foldnum in range(10):
#             for datasetnum in range (295): #5
#                 print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                       '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#                 count+=1
#
# for featsel in ['anova', 'chi2']:
#     for classifier in ['lufe','lufereverse']:
#         for lupimethod in ['svmplus','dp']:
#             for foldnum in range(10):
#                 for datasetnum in range (295): #5
#                     print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                           '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#                     count+=1
#
#     print(count)
#


#
# for classifier in ['featselector','svmreverse']:
#     for featsel in ['anova', 'chi2']:
#         lupimethod = 'nolufe'
#         for foldnum in range(10):
#             for datasetnum in range (295): #5
#                 print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                       '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#                 count+=1
#
# for featsel in ['anova', 'chi2']:
#     for classifier in ['lufe','lufereverse']:
#         for lupimethod in ['svmplus','dp']:
#             for foldnum in range(10):
#                 for datasetnum in range (295): #5
#                     print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                           '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#                     count+=1
#
#     print(count)




# featsel = 'rfe'
# classifier = 'featselector'
# lupimethod = 'nolufe'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1
# print (count)
#
# for featsel in ['mi','rfe']:
#     classifier = 'lufe'
#     lupimethod = 'dp'
#     for foldnum in range(10):
#         for datasetnum in range (295): #5
#             print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                   '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#             count+=1
#
#     classifier = 'lufereverse'
#     lupimethod = 'dp'
#     for foldnum in range(10):
#         for datasetnum in range (295): #5
#             print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#                   '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#             count+=1
# print(count)


# classifier = 'lufe'
# lupimethod = 'svmplus'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1
#
# classifier = 'lufereverse'
# lupimethod = 'svmplus'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1


#
# classifier = 'svmreverse'
# lupimethod = 'nolufe'
# for foldnum in range(10):
#     for datasetnum in range (295): #5
#         print('--foldnum {} --topk {} --dataset {} --datasetnum {} --kernel {} --cmin -3 --cmax 3 --numberofcs 7 --skfseed {} --percentofpriv {} --percentageofinstances {} '
#               '--taketopt {} --lupimethod {} --featsel {} --classifier {}'.format(foldnum, top_k, dataset, datasetnum, 'linear', seed, percentofpriv, 100, take_top_t, lupimethod, featsel, classifier))
#         count+=1
# print(count)
# featsel = 'rfe'



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

