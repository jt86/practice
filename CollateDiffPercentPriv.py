from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all_datasets
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path

topk = 300
lufe_scores = dict()
featsel_scores = dict()

# percentages= [10,25,50,100]
percentages= [10,25,50]#,75]#,100]
featsels = ['rfe','mi'] #''anova','bahsic','chi2','mi','rfe']
for featsel in featsels:
    for percentage in percentages:
        s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                          take_top_t='bottom', lupimethod='svmplus', featsel=featsel, classifier='lufe',
                                percent_of_priv=percentage)
        lufe_scores['{}-{}-{}'.format(featsel, percentage, s1.classifier)] =np.mean(collate_all_datasets(s1,num_datasets=50))

        featsel_scores[featsel]= np.mean(collate_all_datasets(Experiment_Setting(foldnum='all', topk=300, dataset='tech',
                        datasetnum='all', skfseed=1, kernel='linear',take_top_t='top',
                        lupimethod='nolufe', featsel=featsel, classifier='featselector')))
print(lufe_scores.values())

baseline_score = np.mean(collate_all_datasets(Experiment_Setting(foldnum='all', topk='all', dataset='tech',
                        datasetnum='all', skfseed=1, kernel='linear', take_top_t='top', lupimethod='nolufe',
                                                                 featsel='nofeatsel', classifier='baseline')))

for featsel in featsels:
    plt.plot(percentages, [lufe_scores['{}-{}-lufe'.format(featsel,p)] for p in percentages],label='LUFe-{}-SVM+'.format(featsel.upper()))
    plt.axhline(y=featsel_scores[featsel], linestyle='--', label='FeatSel-{}-SVM'.format(featsel.upper()))
plt.legend(loc='best')
plt.xlabel('Percentage of privileged information used')
plt.ylabel('Mean accuracy score (%)')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2c/diffpercentpriv'))
plt.show()

for p in percentages:
    print('{} & {} & {} ')