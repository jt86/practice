from CollateResults import collate_all
from ExperimentSetting import Experiment_Setting
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from Get_Full_Path import get_full_path
def get_scores(featsel, lupimethod,ax):

    svm_reverse_setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                             take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='svmreverse')
    svm_reverse_scores = collate_all(svm_reverse_setting)


    svm_setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                      take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    svm_scores = collate_all(svm_setting)


    lufe_setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                      take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe')
    lufe_scores = collate_all(lufe_setting)

    baseline_setting = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                      take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
    baseline_scores = collate_all(baseline_setting)

    lufe_scores=np.mean(lufe_scores,axis=1)
    svm_scores = np.mean(svm_scores, axis=1)
    baseline_scores=np.mean(baseline_scores,axis=1)
    svm_reverse_scores = np.mean(svm_reverse_scores, axis=1)

    lufe_improvs_vs_all = lufe_scores-baseline_scores
    svm_improvs_vs_all = svm_scores - baseline_scores
    svm_reverse_improvs_vs_all = svm_reverse_scores - baseline_scores
    lufe_improvs_vs_svm = lufe_scores-svm_scores

    improv_diff = (lufe_scores-svm_scores)/(np.array(svm_scores+0.001-baseline_scores))

    ########### COMMENT OUT THESE SO ONLY RUN ONE AT A TIME!
    ########### SVM IMPROVMEMENT VS LUFE IMPROVEMENT

    # x,y = svm_improvs_vs_all,lufe_improvs_vs_svm
    # ax.scatter(x,y,alpha =0.5)
    # ax.set_title('{}: r={:.3f}'.format(featsel.upper(), np.corrcoef(x, y)[0, 1]))
    # ax.set_xlabel('FeatSel vs ALL improvement (%)')
    # ax.set_ylabel('LUFe vs FeatSel improvement (%)')
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    # ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # ax.axvline(x=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # plt.show()

    ########### SVM-REVERSE SCORE VS LUFE IMPROVEMENT

    # x,y = svm_reverse_scores,lufe_improvs_vs_svm
    # ax.scatter(x,y,alpha =0.5)
    # ax.set_title('{}: r={:.3f}'.format(featsel.upper(), np.corrcoef(x, y)[0, 1]))
    # ax.set_xlabel('SVM-Reverse score (%)')
    # ax.set_ylabel('LUFe improvement vs FeatSel (%)')
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    # ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)

    ########### SVM-REVERSE 'IMPROVEMENT' over FeatSel VS LUFE IMPROVEMENT

    # x,y = svm_reverse_scores-svm_scores,lufe_improvs_vs_svm
    # ax.scatter(x,y,alpha =0.5)
    # ax.set_title('{}: r={:.3f}'.format(featsel.upper(), np.corrcoef(x, y)[0, 1]))
    # ax.set_xlabel('SVM-Reverse improvement vs FeatSel (%)')
    # ax.set_ylabel('LUFe improvement vs FeatSel (%)')
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    # ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)

    ########### SVM SCORE VS LUFE IMPROVEMENT

    # x,y = svm_scores,lufe_improvs_vs_svm
    # ax.scatter(x,y,alpha =0.5)
    # ax.set_title('{}: r={:.3f}'.format(featsel.upper(), np.corrcoef(x, y)[0, 1]))
    # ax.set_xlabel('FeatSel accuracy score (%)')
    # ax.set_ylabel('LUFe vs FeatSel improvement (%)')
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    # ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # ax.axvline(x=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # plt.show()

    # x,y = svm_reverse_improvs_vs_all,lufe_improvs_vs_svm
    # ax.scatter(x,y,alpha =0.5)
    # ax.set_title('{}: r={:.3f}'.format(featsel.upper(), np.corrcoef(x, y)[0, 1]))
    # ax.set_xlabel('SVM-Reverse improvement vs ALL (%)')
    # ax.set_ylabel('LUFe vs FeatSel improvement (%)')
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    # ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # ax.axvline(x=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # plt.show()

    #####


    # x,y = svm_reverse_scores,improv_diff
    # ax.scatter(x,y,alpha =0.5)
    # ax.set_title('{}: r={:.3f}'.format(featsel.upper(), np.corrcoef(x, y)[0, 1]))
    # ax.set_xlabel('SVM reverse score (%)')
    # ax.set_ylabel('J score')
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    # ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # ax.axvline(x=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # plt.show()



    print('\n {} {} lufe improvement = {}'.format(featsel,lupimethod,np.mean(lufe_improvs_vs_svm)))


    print('{}.. corr between -> svm reverse, lufe improvement'.format(featsel),np.corrcoef(svm_reverse_improvs_vs_all,lufe_improvs_vs_svm)[0,1])
    print('{}.. corr between -> svm, lufe improvement'.format(featsel), np.corrcoef(svm_scores, lufe_improvs_vs_svm)[0, 1])
    print('{}.. corr between -> lufe, lufe improvement'.format(featsel), np.corrcoef(lufe_scores, lufe_improvs_vs_svm)[0, 1])

    print('{}.. corr between -> svm reverse, lufe_scores'.format(featsel),np.corrcoef(svm_reverse_scores,lufe_scores)[0,1])
    print('{}.. corr between -> svm, lufe_scores'.format(featsel), np.corrcoef(svm_scores, lufe_scores)[0, 1])
    print('{}.. corr between -> lufe, lufe_scores'.format(featsel), np.corrcoef(lufe_scores, lufe_scores)[0, 1])

#     # np.corrcoef()

plt.clf()
fig = plt.figure(figsize=[7,9])
for lupimethod in ['svmplus']:#,'dp']:
    print('\n LUPI method',lupimethod)
    # for featsel in ['mi']:
    for i, featsel in enumerate(['anova','bahsic','chi2','mi','rfe']):
        print('\n Feature selection method', featsel, i)
        get_scores(featsel, lupimethod, ax=fig.add_subplot(3,2,i+1))

plt.subplots_adjust(top=0.95, bottom=0.05,hspace=0.35,wspace=0.35)
# plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/svmreverse_improv_vs_lufe_improv.pdf'),format='pdf')
# plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/svm_reverse_vs_j_score2.pdf'),format='pdf')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/svmreverse_improv_over_featsel_vs_lufe_improv.pdf'),format='pdf')
# plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/svm_score_vs_lufe_improv.pdf'),format='pdf')
plt.show()