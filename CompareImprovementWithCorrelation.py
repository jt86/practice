from CollateResults import collate_all_datasets
from SingleFoldSlice import Experiment_Setting
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from Get_Full_Path import get_full_path
def get_scores(featsel, lupimethod):

    svm_reverse_setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                             take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='svmreverse')
    svm_reverse_scores = collate_all_datasets(svm_reverse_setting)


    svm_setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                      take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    svm_scores = collate_all_datasets(svm_setting)


    lufe_setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                      take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe')
    lufe_scores = collate_all_datasets(lufe_setting)

    baseline_setting = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                      take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
    baseline_scores = collate_all_datasets(baseline_setting)

    lufe_scores=np.mean(lufe_scores,axis=1)
    svm_scores = np.mean(svm_scores, axis=1)
    baseline_scores=np.mean(baseline_scores,axis=1)
    svm_reverse_scores = np.mean(svm_reverse_scores, axis=1)
    lufe_scores -= baseline_scores
    svm_scores -= baseline_scores
    svm_reverse_scores-=baseline_scores

    lufe_improvements = (lufe_scores - svm_scores) * 100

    # plt.scatter(svm_reverse_scores,lufe_improvement)
    # plt.show()
    print('\n {} {} lufe improvement = {}'.format(featsel,lupimethod,np.mean(lufe_improvements)))




    list1  = [baseline_scores,svm_scores,lufe_scores,svm_reverse_scores,lufe_improvements]
    list_of_corrs = np.zeros((len(list1), len(list1)))
    for ind1,item1 in enumerate(list1):
        for ind2,item2 in enumerate(list1):
              list_of_corrs[ind1,ind2]=(round(np.corrcoef(item1, item2)[0, 1],4))#'{} \t'.format(round(np.corrcoef(item1, item2)[0, 1],4)))

    # plt.imshow(list_of_corrs, cmap='hot', interpolation='nearest')

    # plt.xlabel('baseline, svm, lufe, svm reverse')
    # plt.show()
    list_of_labels=['ALL feats\n baseline SVM', 'Feature \n selection SVM', 'LUFe', 'Unselected \nonly svm', 'LUFe v featsel \nimprovement']
    sns.heatmap(list_of_corrs,xticklabels=list_of_labels,yticklabels=list_of_labels)
    plt.title('Correlations in accuracy score \n Feat selection:{}, LUPI method:{}'.format(featsel,lupimethod))
    plt.savefig(get_full_path('Desktop/Privileged_Data/CorrelationPlots/{}-{}'.format(lupimethod,featsel)))
    print(list_of_corrs)
    plt.clf()

    # print('{}.. corr between -> svm reverse, lufe improvement'.format(featsel),np.corrcoef(svm_reverse_scores,lufe_improvement)[0,1])
    # print('{}.. corr between -> svm, lufe improvement'.format(featsel), np.corrcoef(svm_scores, lufe_improvement)[0, 1])
    # print('{}.. corr between -> lufe, lufe improvement'.format(featsel), np.corrcoef(lufe_scores, lufe_improvement)[0, 1])

    # print('{}.. corr between -> svm reverse, lufe_scores'.format(featsel),np.corrcoef(svm_reverse_scores,lufe_scores)[0,1])
    # print('{}.. corr between -> svm, lufe_scores'.format(featsel), np.corrcoef(svm_scores, lufe_scores)[0, 1])
    # print('{}.. corr between -> lufe, lufe_scores'.format(featsel), np.corrcoef(lufe_scores, lufe_scores)[0, 1])

    # np.corrcoef()

for lupimethod in ['svmplus','dp']:
    print('\n LUPI method',lupimethod)
    for featsel in ['rfe','anova','chi2','bahsic','mi']:
        print('\n Feature selection method', featsel)
        get_scores(featsel, lupimethod)

