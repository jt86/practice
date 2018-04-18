from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all_datasets
import numpy as np
from matplotlib import pyplot as plt

baseline_setting = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
baseline = np.mean(collate_all_datasets(baseline_setting),axis=1)

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
    s1_means = np.mean(collate_all_datasets(s1),axis=1)
    s2_means = np.mean(collate_all_datasets(s2), axis=1)
    s1_diffs = s1_means - baseline
    s2_diffs = s2_means - baseline
    # print('s1',(s1_means-baseline), 's2',s2_means-baseline)
    scores[featsel]=[np.mean(s1_diffs), np.mean(s2_diffs)]
    stdevs[featsel]=[np.std(s1_diffs),np.std(s2_diffs)]


# print(len(collate_all_datasets(s1) - baseline))

# print(stdevs['anova'])
print(np.mean(collate_all_datasets(s1),axis=1))#.shape)
print(np.std(np.mean(collate_all_datasets(s1),axis=1)))


# univariate_results['ALL'] = [np.mean(collate_all_datasets(baseline))]


# plotting UNIVARIATE feature selection and LUFe results
fig = plt.figure(figsize=(10, 4))
fig, ax = plt.subplots()

print([scores[key][0] for key in featsels])
print(featsels)
print(stdevs[key][0] for key in featsels)
bar_width=0.35

indices = np.array(range(len(featsels)))
bar_width = 0.35


#yerr=[stdevs[key][0] for key in featsels]
#yerr = [stdevs[key][1] for key in featsels]

plt.bar(indices, [scores[key][0] for key in featsels], bar_width,yerr=[stdevs[key][0] for key in featsels])
plt.bar(indices + bar_width, [scores[key][1] for key in featsels], bar_width, yerr = [stdevs[key][1] for key in featsels])
# ax.set_xticklabels(['']+featsels)
plt.xticks(indices + bar_width / 2, [i.upper() for i in featsels])
plt.xlabel('Feature selection metric')
plt.ylabel('Improvement over all-feature baseline (%)')
# plt.ylim((80,87))
plt.show()