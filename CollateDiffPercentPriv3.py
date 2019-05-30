from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
import pickle


topk = 300


percentages= [10,25,50,100]
percentages= [10,25,50,75,100]
featsels = ['rfe','mi']

lufe_scores = dict()
featsel_scores = dict()

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


# print(len(lufe_scores), [len(lufe_scores[i]) for i in lufe_scores])
# #
# for key in lufe_scores:
#     lufe_scores[key] = np.mean(lufe_scores[key])
#
# for key in featsel_scores:
#     featsel_scores[key] = np.mean(featsel_scores[key])
#
#
# with open("lufe_scores.pkl","wb") as f:
#     pickle.dump(lufe_scores,f)
#
# with open("featsel_scores.pkl","wb") as f:
#     pickle.dump(featsel_scores,f)
#
# with open("baseline_score.pkl","wb") as f:
#     pickle.dump(baseline_score,f)
#
avg_lufe_scores, avg_featsel_scores = dict(), dict()

for key in lufe_scores:
    avg_lufe_scores[key] = np.mean(lufe_scores[key])

for key in featsel_scores:
    avg_featsel_scores[key] = np.mean(featsel_scores[key])


    # print(type(lufe_scores), type(baseline_score))
fig, axes = plt.subplots(2,1, figsize=[6,10])
for i, featsel in enumerate(featsels):
    ax = axes.flat[i]
    x_values = np.array(range(5))
    for j, take_top_t in enumerate(['top','bottom']):
        x_values2 = (x_values-0.2 if take_top_t=='top' else x_values+0.2)
        print(x_values2)
        scores = [avg_lufe_scores['{}-{}-{}-lufe'.format(featsel, take_top_t, p)] for p in percentages]
        yerr = [np.std(lufe_scores['{}-{}-{}-lufe'.format(featsel, take_top_t, p)]) for p in percentages]
        print('yerr', yerr)
        ax.bar(x_values2, scores, label='LUFe-{}-SVM+ ({})'.format(featsel.upper(),take_top_t), width=0.4, yerr = yerr)
    ax.axhline(y=avg_featsel_scores[featsel], linestyle='--', label='FeatSel-{}-SVM'.format(featsel.upper()), c='k')
    ax.legend(loc='upper right')
    ax.set_xticklabels([0]+percentages)
    ax.set_xlabel('Percentage of privileged information used')
    ax.set_ylabel('Mean accuracy score (%)')
    ax.set_ylim(bottom=80, top=89)
    ax.set_title(featsel.upper())
plt.subplots_adjust(hspace=0.3)
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2c/diffpercentpriv.pdf'),type='pdf')
plt.show()

for p in percentages:
    print('{}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% \\\\'
          .format(p,lufe_scores['rfe-top-{}-lufe'.format(p)], lufe_scores['mi-top-{}-lufe'.format(p)],
        lufe_scores['rfe-bottom-{}-lufe'.format(p)],lufe_scores['mi-bottom-{}-lufe'.format(p)]))