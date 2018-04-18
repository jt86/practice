from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all_datasets
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path

# collating UNIVARIATE feature selection and LUFe results for different feature selection
topk = 300
univariate_results = dict()
featsels = ['anova','bahsic','chi2','mi']
for featsel in featsels:
    s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    s2 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
    univariate_results[featsel]=[np.mean(collate_all_datasets(s1)), np.mean(collate_all_datasets(s2))]
baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')

# collating RFE feature selection and LUFe results for different stepsizes
stepsizes = [0.5,0.1,0.01,0.001]
rfe_fs_results = np.zeros(len(stepsizes))
rfe_lufe_results = np.zeros(len(stepsizes))
for count,item in enumerate(stepsizes):
    s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='nolufe', featsel='rfe', classifier='featselector', stepsize=item)
    s2 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                      take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='lufe', stepsize=item)
    rfe_fs_results[count]=np.mean(collate_all_datasets(s1))
    rfe_lufe_results[count]= np.mean(collate_all_datasets(s2))


print(rfe_lufe_results)
print(rfe_fs_results)

# plotting UNIVARIATE feature selection and LUFe results
fig = plt.figure(figsize=(10, 4))
rfe_settings = [rfe_fs_results,rfe_lufe_results]
titles = ['Standard Feature Selection','LUFe']
for i in range(2):
    fig1 = plt.subplot(1, 2, i+1)
    for item in featsels:
        fig1.plot(stepsizes, [univariate_results[item][i]] * len(stepsizes), '-', label=item.upper())
    fig1.plot(stepsizes,rfe_settings[i],'ko--', label='RFE')
    plt.ylabel('Mean accuracy (%)')
    plt.xlabel('Stepsize for RFE')
    plt.xscale('log')
    plt.title(titles[i])
    if i==1:
        plt.legend(loc='lower right')
    fig1.ylim=([81,87])
    plt.gca().set_ylim([81, 87])
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2a/stepsizeplot'))
plt.show()
