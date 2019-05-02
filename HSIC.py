import time
import numpy as np
import tensorflow as tf
from SingleFoldSlice import get_norm_priv
from GetSingleFoldData import get_train_and_test_this_fold
from ExperimentSetting import Experiment_Setting
from Get_Full_Path import get_full_path
import csv
from CollateResults import collate_single_dataset
from matplotlib import pyplot as plt

def quadratic_time_HSIC(data_first, data_second, sigma):
    XX = np.dot(data_first, data_first.transpose())
    YY = np.dot(data_second, data_second.transpose())
    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    r = lambda x: np.expand_dims(x, 0)
    c = lambda x: np.expand_dims(x, 1)

    gamma = 1 / (2 * sigma ** 2)
    # use the second binomial formula
    Kernel_XX = np.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
    Kernel_YY = np.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))

    Kernel_XX_mean = np.mean(Kernel_XX, 0)
    Kernel_YY_mean = np.mean(Kernel_YY, 0)

    HK = Kernel_XX - Kernel_XX_mean
    HL = Kernel_YY - Kernel_YY_mean

    n = Kernel_YY.shape[0]
    HKf = HK / (n - 1)
    HLf = HL / (n - 1)

    # biased estimate
    hsic = np.trace(np.dot(HKf.transpose(), HLf))
    return hsic

# data_first = np.random.rand(3,3)
# data_second = np.random.rand(3,3)
# print(data_first)
# print(quadratic_time_HSIC(data_first,data_second,1))



###### THIS PART WRITES THE HSIC RESULTS
#


def write_hsic_results(name, featsel='mi', percent_of_priv=100, classifier='lufe', lupimethod='svmplus'):
    assert name in ['normal-with-labels','priv-with-labels', 'normal-with-priv']
    with open(get_full_path('Desktop/Privileged_Data/HSICdependencies/HSIC-{}-{}-{}.csv'.format(name,featsel,percent_of_priv)), 'a') as results_file:
        results_writer = csv.writer(results_file)
        for datasetnum in range(295):
            for foldnum in range(10):
                s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum,
                                       kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=percent_of_priv,
                                       percentageofinstances=100, take_top_t='top', lupimethod=lupimethod,
                                       featsel=featsel, classifier=classifier, stepsize=0.1)
                all_train, all_test, labels_train, labels_test = get_train_and_test_this_fold(s)
                normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)
                labels_train2 = labels_train.reshape(len(labels_train), 1)
                if name == 'normal-with-labels':
                    results_writer.writerow([datasetnum, foldnum, (quadratic_time_HSIC(normal_train, labels_train2, sigma=1))])
                if name == 'priv-with-labels':
                    results_writer.writerow([datasetnum, foldnum, (quadratic_time_HSIC(priv_train, labels_train2, sigma=1))])
                if name == 'normal-with-priv':
                    results_writer.writerow([datasetnum, foldnum, (quadratic_time_HSIC(normal_train, priv_train, sigma=1))])

# write_hsic_results('normal-with-priv','mi')


def get_hsic(s,name):
    list_of_hsics=np.zeros(10)
    with open(get_full_path('Desktop/Privileged_Data/HSICdependencies/HSIC-{}-{}-{}.csv'.format(name,s.featsel,s.percent_of_priv)), 'r') as results_file:
        results_reader = csv.reader(results_file)
        for line in results_reader:
            #10 folds - use index to put fold result in the right place
            if int(line[0])==s.datasetnum:
                list_of_hsics[int(line[1])]=line[2]
            if 0 not in list_of_hsics:
                return(list_of_hsics)


def compare_hsic_w_improvement(name, ax, featsel='mi', percent_of_priv=100, classifier='lufe', lupimethod='svmplus'):

    improvements, hsics  = [], []

    for datasetnum in range(295):

        s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum=datasetnum,
                               kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=1,
                               percent_of_priv=percent_of_priv,
                               percentageofinstances=100, take_top_t='top', lupimethod='nolufe',
                               featsel='nofeatsel', classifier='baseline', stepsize=0.1)

        s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum,
                               kernel='linear', cmin=-3, cmax=3, numberofcs=7, skfseed=1,
                               percent_of_priv=percent_of_priv,
                               percentageofinstances=100, take_top_t='top', lupimethod=lupimethod,
                               featsel=featsel, classifier=classifier, stepsize=0.1)

        hsic_list = get_hsic(s, name)
        accuracy_list = collate_single_dataset(s)
        baseline_accuracy_list = collate_single_dataset(s_baseline)
        hsic = np.mean(hsic_list)
        improvement = (np.mean(accuracy_list)-np.mean(baseline_accuracy_list))*100
        improvements.append(improvement)
        hsics.append(hsic)
        print('improvement', improvement)
        print(datasetnum)
    x,y = hsics, improvements

    plt.xscale('log')
    ax.scatter(x,y,alpha=0.2)
    # ax.plot(x, np.poly1d(np.polyfit(x, y, 1))(x), color='k')
    ax.set_title('{}: r={:.3f}'.format(featsel.upper(),np.corrcoef(hsics,improvements)[0,1]))
    xlabel_dict = {'normal-with-labels':'HSIC between $\mathcal{\widehat{S}}$ and $\it{y}$',
                   'priv-with-labels':'HSIC between $\mathcal{\widehat{U}}$ and $\it{y}$',
                   'normal-with-priv':'HSIC between $\mathcal{\widehat{S}}$ and $\mathcal{\widehat{U}}$'}
    ax.set_ylabel('LUFe improvement vs FeatSel (%)')
    ax.set_xlabel(xlabel_dict[name])
    ax.set_xticks([10**-3, 10**-2, 10**-1])
    ax.axhline(y=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    # ax.axvline(x=0, clip_on=False, linestyle='dashed', color='k', lw=0.5)
    return(np.corrcoef(hsics,improvements)[0,1])

# for featsel in ['anova','bahsic','chi2','rfe']:
#     write_hsic_results('normal-with-labels',featsel)
#     write_hsic_results('priv-with-labels', featsel)
#     write_hsic_results('normal-with-priv', featsel)

# for featsel in ['anova', 'bahsic', 'chi2', 'rfe']:
#     compare_hsic_w_improvement('normal-with-labels', featsel='mi', percent_of_priv=100, classifier='lufe', lupimethod='svmplus')
#     compare_hsic_w_improvement('normal-with-labels', featsel='mi', percent_of_priv=100, classifier='lufe',
#                                lupimethod='svmplus')
#     compare_hsic_w_improvement('normal-with-labels', featsel='mi', percent_of_priv=100, classifier='lufe',
#                                lupimethod='svmplus')




percent_of_priv = 100
name = 'normal-with-labels'

all_hsics = np.zeros([5,3])


for j, name in enumerate(['normal-with-labels','priv-with-labels','normal-with-priv']):
    fig = plt.figure(figsize=[7, 9])
    for i, featsel in enumerate(['anova', 'bahsic', 'chi2', 'mi', 'rfe']):
        all_hsics[i, j] = compare_hsic_w_improvement(name, ax=fig.add_subplot(3,2,i+1), featsel=featsel, percent_of_priv=100, classifier='lufe', lupimethod='svmplus')
        plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.35, wspace=0.35)
        plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/HSIC-correlations/HSIC-{}-{}.pdf'.format(name, percent_of_priv)),format='pdf')
#
# for item in all_hsics:
#     print(' '.join(['& {:.2f}'.format(a) for a in item]))

        # plt.show()