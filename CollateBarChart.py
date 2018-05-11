from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from Get_Full_Path import get_full_path

# collating all results for different feature selection
topk = 300
scores = dict()
stdevs = dict()
featsels = ['anova','bahsic','chi2','mi','rfe']
for featsel in featsels:
    s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    s2 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
    scores[featsel]=[np.mean(collate_all(s1)), np.mean(collate_all(s2))]
    stdevs[featsel]=[np.std(np.mean(collate_all(s1), axis=1)), np.std(collate_all(s2))]

print(stdevs['anova'])
print(np.mean(collate_all(s1), axis=1))#.shape)

baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
baseline_score = np.mean(collate_all(baseline))
# univariate_results['ALL'] = [np.mean(collate_all_datasets(baseline))]


# plotting UNIVARIATE feature selection and LUFe results
# fig = plt.figure(figsize=(10, 4))
fig, ax = plt.subplots(figsize = (7,7))
# ax = Axes(fig=fig,rect=)
# baseline_line = ax.axhline(y=baseline_score)

indices = np.array(range(len(featsels)))
indices2 = [indices]
print (indices,indices2)
bar_width = 0.35
#yerr=[stdevs[key][0] for key in featsels]
#yerr = [stdevs[key][1] for key in featsels]

plt.subplot(2,1,1)
# plt.plot(,[baseline_score]*len(indices), 'r-.')
baseline_line = plt.axhline(y=baseline_score,c='k',linestyle='--',label='ALL')
plt.bar(indices, [scores[key][0] for key in featsels], bar_width, label = 'FeatSel')
plt.bar(indices + bar_width, [scores[key][1] for key in featsels], bar_width, label = 'LUFe')
plt.xticks(indices + bar_width / 2, [i.upper() for i in featsels])
# plt.tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
plt.ylabel('Mean accuracy (%)')
plt.ylim((80,87))
plt.legend()#loc='lower right')
plt.subplot(2,1,2)
plt.bar(indices+ bar_width / 2, [scores[key][1] - scores[key][0] for key in featsels], bar_width, color='g')
plt.xticks(indices + bar_width / 2, [i.upper() for i in featsels])
plt.xlabel('Feature selection metric')
plt.ylabel('Mean improvement by LUFe (%)')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2a/improvementsbar'))
plt.show()