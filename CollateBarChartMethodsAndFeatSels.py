from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all_datasets
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
# collating all results for different feature selection
topk = 300
scores = dict()
stdevs = dict()
featsels = ['anova','bahsic','chi2','mi','rfe']
methods = ['svmplus', 'dsvm','dp']

for featsel in featsels:
    s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    scores['{}-featsel'.format(featsel)] = np.mean(collate_all_datasets(s1))
    for method in methods:
        s2 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                          take_top_t='top', lupimethod=method, featsel=featsel, classifier='lufe')
        scores['{}-{}'.format(featsel,method)]=np.mean(collate_all_datasets(s2))

print(scores.keys())


baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
baseline_score = np.mean(collate_all_datasets(baseline))


names_dict = {'svmplus':'SVM+','dsvm':'dSVM','dp':r'SVM$\delta$+','featsel':'FeatSel'}

fig, ax = plt.subplots(figsize = (7,7))

indices = np.array(range(len(featsels)))
bar_width = 0.2
print(scores['{}-featsel'.format(featsel)])
plt.axhline(y=baseline_score,c='k',linestyle='--',label='ALL')
for i, method in enumerate(methods+['featsel']):
    print([scores['{}-{}'.format(f,method)] for f in featsels])
    plt.bar(indices+(i*bar_width), [scores['{}-{}'.format(f,method)] for f in featsels], bar_width, label=names_dict[method])
plt.ylim((78,87))
plt.xticks(indices + bar_width*1.5, [i.upper() for i in featsels])
plt.legend()
plt.xlabel('Feature Selection method')
plt.ylabel('Accuracy score (%)')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2b/methodandfeatselbar'))
plt.show()


