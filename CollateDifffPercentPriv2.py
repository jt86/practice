from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
import pickle

topk = 300

percentages = [10, 25, 50, 100]
percentages = [10, 25, 50, 75, 100]
featsels = ['rfe', 'mi']  # ''anova','bahsic','chi2','mi','rfe']
#
# lufe_scores = dict()
# featsel_scores = dict()
#
# for featsel in featsels:
#     for take_top_t in ['top', 'bottom']:
#         for percentage in percentages:
#             s1 = Experiment_Setting(foldnum='all', topk=topk, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
#                                               take_top_t=take_top_t, lupimethod='svmplus', featsel=featsel, classifier='lufe',
#                                     percent_of_priv=percentage)
#             lufe_scores['{}-{}-{}-{}'.format(featsel, s1.take_top_t, percentage, s1.classifier)] = collate_all(s1, num_datasets=295)
#
#             featsel_scores[featsel]= collate_all(Experiment_Setting(foldnum='all', topk=300, dataset='tech',
#                                                                             datasetnum='all', skfseed=1, kernel='linear', take_top_t='top',
#                                                                             lupimethod='nolufe', featsel=featsel, classifier='featselector'))
# print(lufe_scores.values())
#
# baseline_score = collate_all(Experiment_Setting(foldnum='all', topk='all', dataset='tech',
#                                                         datasetnum='all', skfseed=1, kernel='linear', take_top_t='top', lupimethod='nolufe',
#                                                         featsel='nofeatsel', classifier='baseline'))
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




with open("lufe_scores.pkl", 'rb') as f:
    lufe_scores = pickle.load(f)

with open("featsel_scores.pkl", 'rb') as f:
    featsel_scores = pickle.load(f)

with open("baseline_score.pkl", 'rb') as f:
    baseline_score = pickle.load(f)

with open("lufe_scores.pkl","wb") as f:
    pickle.dump(lufe_scores,f)

with open("featsel_scores.pkl","wb") as f:
    pickle.dump(featsel_scores,f)

with open("baseline_score.pkl","wb") as f:
    pickle.dump(baseline_score,f)




with open("lufe_scores.pkl", 'rb') as f:
    lufe_scores = pickle.load(f)

with open("featsel_scores.pkl", 'rb') as f:
    featsel_scores = pickle.load(f)

with open("baseline_score.pkl", 'rb') as f:
    baseline_score = pickle.load(f)


print(type(lufe_scores), type(baseline_score))
fig, axes = plt.subplots(2, 1)#, sharex=True)
for i, featsel in enumerate(featsels):
    for j, take_top_t in enumerate(['top', 'bottom']):
        ax = axes[i]
        scores  = [lufe_scores['{}-{}-{}-lufe'.format(featsel, take_top_t, p)] for p in percentages]
        print(np.mean(scores))

        print(featsel, take_top_t, [len(a) for a in scores])
        print('xticks', np.array(range(len(scores))) - 0.5 + j)
        ax.bar(np.array(range(len(scores)))-0.5+j, [np.mean(a) for a in scores], width=0.5)
        ax.set_xticklabels(percentages)

# print(type(lufe_scores), type(baseline_score))
# fig, axes = plt.subplots(2, 2, sharey=True)
# for i, featsel in enumerate(featsels):
#     for j, take_top_t in enumerate(['top', 'bottom']):
#         ax = axes[i,j]
#         scores  = [lufe_scores['{}-{}-{}-lufe'.format(featsel, take_top_t, p)] for p in percentages]
#         print(np.mean(scores))
#         print(featsel, take_top_t, [len(a) for a in scores])
#         ax.boxplot(scores, showfliers=False)

#     ax.axhline(y=featsel_scores[featsel], linestyle='--', label='FeatSel-{}-SVM'.format(featsel.upper()), c='k')
#     ax.legend(loc='best')
#     ax.set_xlabel('Percentage of privileged information used')
#     ax.set_ylabel('Mean accuracy score (%)')
#     ax.set_title(featsel.upper())
#     # plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap2c/diffpercentpriv.pdf'),type='pdf')
plt.show()
#
# for p in percentages:
#     print('{}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% \\\\'
#           .format(p, lufe_scores['rfe-top-{}-lufe'.format(p)], lufe_scores['mi-top-{}-lufe'.format(p)],
#                   lufe_scores['rfe-bottom-{}-lufe'.format(p)], lufe_scores['mi-bottom-{}-lufe'.format(p)]))