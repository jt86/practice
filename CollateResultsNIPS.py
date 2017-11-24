from Get_Full_Path import get_full_path
from SingleFoldSlice import make_directory, Experiment_Setting
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mutual_info_score
from SingleFoldSlice import get_train_and_test_this_fold, get_norm_priv

def collate_single_dataset(s):
    results=np.zeros(10)
    output_directory = get_full_path(('Desktop/Privileged_Data/NIPSResults/{}/{}{}/').format(s.name,s.dataset, s.datasetnum))
    assert os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed))),'{} does not exist'.format(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
    with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
        for item in csv.reader(cv_lupi_file):
            results[item[0]]=item[1]
    assert 0 not in results
    if 0 in results:
        print("print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top')"
        .format(np.where(results == 0)[0][0], 300, 'tech', s.datasetnum, 1,  s.lupimethod, s.featsel, s.classifier, s.stepsize, s.percentageofinstances))
    return results

all_results = []




#
# def get_all_results(dataset):
#     s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
#              cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='nolufe',
#              featsel='nofeatsel',classifier='baseline',stepsize=0.1)
#     baseline_score= (np.mean(collate_single_dataset(s)))
#
#     for featsel in ['rfe', 'chi2', 'anova']:#, 'mi']:
#         s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
#                  cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='nolufe',
#                  featsel=featsel,classifier='featselector',stepsize=0.1)
#         featsel_score = (np.mean(collate_single_dataset(s)))
#         print(featsel_score)
#
#         s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
#                  cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus',
#                  featsel=featsel,classifier='lufe',stepsize=0.1)
#         lufe_score = (np.mean(collate_single_dataset(s)))
#
#         print(s.dataset,'\t',s.featsel,'\t',baseline_score,'\t',featsel_score,'\t',lufe_score)


def get_all_results(dataset):
    # s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
    #          cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='nolufe',
    #          featsel='nofeatsel',classifier='baseline',stepsize=0.1)
    # baseline_score= (np.mean(collate_single_dataset(s)))
    scores1, scores2 =[],[]
    for topk in range(10,100,10):
        s1 = Experiment_Setting(foldnum='all', topk=topk, dataset=dataset, datasetnum=0, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus',
                 featsel='rfe',classifier='lufe',stepsize=0.1)

        s2 = Experiment_Setting(foldnum='all', topk=topk, dataset=dataset, datasetnum=0, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus',
                 featsel='rfe',classifier='lufe',stepsize=0.1)

        featsel_score = (np.mean(collate_single_dataset(s1)))
        print(featsel_score)
        lufe_score = (np.mean(collate_single_dataset(s2)))
        print(lufe_score)

        # s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
        #          cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus',
        #          featsel=featsel,classifier='lufe',stepsize=0.1)
        # lufe_score = (np.mean(collate_single_dataset(s)))

        print(s1.dataset,'\t',s1.featsel)#,'\t',baseline_score,'\t')
        scores1.append(featsel_score)
        scores2.append(lufe_score)
    plt.plot(scores1)
    plt.plot(scores2)
    plt.title('')

    plt.show()

for dataset in ['arcene', 'dexter', 'dorothea', 'madelon']:
    print('\n ',dataset)
    get_all_results(dataset)


# classifier = 'baseline'
# lupimethod = 'nolufe'
# featsel = 'nofeatsel'
# print ('-----{}, {}, {} ------'.format(classifier,lupimethod,featsel))
# results=[]
# for dataset in ['arcene','dexter','dorothea','madelon']:
#     s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
#          cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod=lupimethod,
#          featsel=featsel,classifier=classifier,stepsize=0.1)
#     print(dataset,np.mean(collate_single_dataset(s)))
#     results.append(np.mean(collate_single_dataset(s)))
#
#
# lupimethod = 'nolufe'
# print ('-----{}, {}, {} ------'.format(classifier,lupimethod,featsel))
# results=[]
# for featsel in ['rfe','anova','chi2','mi']:#,'bahsic']:
#     for dataset in ['arcene','dexter','dorothea','madelon']:
#
#         s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
#              cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod=lupimethod,
#              featsel=featsel,classifier='featselector',stepsize=0.1)
#         print(dataset,np.mean(collate_single_dataset(s)))
#         results.append(np.mean(collate_single_dataset(s)))
#
#         s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum=0, kernel='linear',
#                                cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                                take_top_t='top', lupimethod='svmplus',
#                                featsel=featsel, classifier='lufe', stepsize=0.1)
#         print(dataset, np.mean(collate_single_dataset(s)))
#         results.append(np.mean(collate_single_dataset(s)))