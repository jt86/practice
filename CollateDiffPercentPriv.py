from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path

topk = 300
lufe_scores = dict()
featsel_scores = dict()

# percentages= [10,25,50,100]
percentages= [10,25,50,75,100]
featsels = ['rfe','mi'] #''anova','bahsic','chi2','mi','rfe']
for featsel in featsels:
    for take_top_t in ['top', 'bottom']:
        for percentage in percentages:
            s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                              take_top_t=take_top_t, lupimethod='svmplus', featsel=featsel, classifier='lufe',
                                    percent_of_priv=percentage)
            lufe_scores['{}-{}-{}-{}'.format(featsel, s1.take_top_t, percentage, s1.classifier)] =np.mean(collate_all(s1, num_datasets=295))

            featsel_scores[featsel]= np.mean(collate_all(Experiment_Setting(foldnum='all', topk=300, dataset='tech',
                                                                            datasetnum='all', skfseed=1, kernel='linear', take_top_t='top',
                                                                            lupimethod='nolufe', featsel=featsel, classifier='featselector')))
print(lufe_scores.values())

baseline_score = np.mean(collate_all(Experiment_Setting(foldnum='all', topk='all', dataset='tech',
                                                        datasetnum='all', skfseed=1, kernel='linear', take_top_t='top', lupimethod='nolufe',
                                                        featsel='nofeatsel', classifier='baseline')))

for featsel in featsels:
    for take_top_t in ['top','bottom']:
        plt.plot(percentages, [lufe_scores['{}-{}-{}-lufe'.format(featsel,take_top_t,p)] for p in percentages],
                 marker='o',label='LUFe-{}-SVM+ ({})'.format(featsel.upper(),take_top_t))
plt.axhline(y=featsel_scores['rfe'], linestyle=':', label='FeatSel-RFE-SVM', c='k')
plt.axhline(y=featsel_scores['mi'], linestyle='--', label='FeatSel-MI-SVM', c='k')
plt.legend(loc='best')
plt.xlabel('Percentage of privileged information used')
plt.ylabel('Mean accuracy score (%)')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2c/diffpercentpriv'))
# plt.show()

for p in percentages:
    print('{}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% \\\\'
          .format(p,lufe_scores['rfe-top-{}-lufe'.format(p)], lufe_scores['mi-top-{}-lufe'.format(p)],
        lufe_scores['rfe-bottom-{}-lufe'.format(p)],lufe_scores['mi-bottom-{}-lufe'.format(p)]))