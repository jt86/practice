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
featsel, classifier, lupimethod = 'rfe', 'featselector', 'nolufe'
with open(get_full_path('Desktop/Privileged_Data/HSICdependencies/HSIC-privtop10withnorm.csv'), 'a') as results_file:
    results_writer = csv.writer(results_file)
    for datasetnum in range(295):
        for foldnum in range(10):
            s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                                           cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=10, percentageofinstances=100,
                                           take_top_t='top', lupimethod=lupimethod,
                                           featsel=featsel, classifier=classifier, stepsize=0.1)
            all_train, all_test, labels_train, labels_test = get_train_and_test_this_fold(s)
            normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)
            labels_train2=labels_train.reshape(len(labels_train),1)
            print('aaaa \n',normal_train.shape,priv_train.shape, labels_train2.shape)
            results_writer.writerow([datasetnum,foldnum,(quadratic_time_HSIC(labels_train2, normal_train, sigma=1))])



###### THIS PART READS AND PLOTS THE HSIC RESULTS

# def compare_hsic_w_improvement(s):
#     accuracy = collate_single_dataset(s)[s.foldnum]
#     with open(get_full_path('Desktop/Privileged_Data/HSICdependencies/HSIC.csv'), 'r') as results_file:
#         results_reader = csv.reader(results_file)
#         for line in results_reader:
#             if int(line[0])==s.datasetnum and int(line[1])==s.foldnum:
#                 return(accuracy,float(line[-1]))



def compare_hsic_w_improvement2(s):
    list_of_hsics=np.zeros(10)
    accuracy = collate_single_dataset(s)
    with open(get_full_path('Desktop/Privileged_Data/HSICdependencies/HSIC-privtop10withnorm.csv'), 'r') as results_file:
        results_reader = csv.reader(results_file)
        for line in results_reader:
            if int(line[0])==s.datasetnum:
                list_of_hsics[int(line[1])]=line[2]
                # print(line[1])
            if 0 not in list_of_hsics:
                return(accuracy,list_of_hsics)




featsel, classifier, lupimethod = 'mi', 'featselector', 'nolufe'
accuracies, hsics = [],[]
for datasetnum in range(295):
    for foldnum in range(10):
        s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                                       cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=10, percentageofinstances=100,
                                       take_top_t='top', lupimethod=lupimethod,
                                       featsel=featsel, classifier=classifier, stepsize=0.1)
    accuracy,hsic=(compare_hsic_w_improvement2(s))
    accuracies.append(np.mean(accuracy))
    hsics.append(np.mean(hsic))

print(len(accuracies))
print(len(hsics))
fig,ax=plt.subplots()
plt.scatter(hsics,accuracies, alpha=0.5)
plt.xlabel('hsics')
plt.ylabel('accuracies')
ax.set_xlim(0,0.02)
plt.show()

print(np.corrcoef(hsics,accuracies))