from ExperimentSetting import Experiment_Setting
from CollateResults import collate_single_dataset
from matplotlib import pyplot as plt
classifier = 'featselector'
lupimethod = 'nolufe'
featsel='rfe'
import numpy as np

def collate_learning_curve(datasetnum):
    means_featsel, means_lufe, mean_self_svm, mean_self_lufe = [],[],[],[]
    for instances in [10,20,30,40,50,60,70,80,90]:
        setting_featsel = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod='nolufe',
                 featsel=featsel,classifier='featselector',stepsize=0.1)
        means_featsel.append(np.mean(collate_single_dataset(setting_featsel)))

        setting_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod='svmplus',
                 featsel=featsel,classifier='lufe',stepsize=0.1)
        means_lufe.append(np.mean(collate_single_dataset(setting_lufe)))

        setting_self_svm = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                 cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod='nolufe',
                 featsel=featsel,classifier='svmtrain',stepsize=0.1)
        mean_self_svm.append(np.mean(collate_single_dataset(setting_self_svm)))

        # setting_self_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
        #          cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod='svmplus',
        #          featsel=featsel,classifier='lufetrain',stepsize=0.1)
        # means_lufe.append(np.mean(collate_single_dataset(setting_self_lufe)))


    print(means_featsel)
    print(means_lufe)
    print(mean_self_svm)

    plt.plot(range(10,100,10),[1-item for item in means_featsel], label='RFE-SVM on testing data')
    plt.plot(range(10,100,10), [1-item for item in means_lufe], label='RFE-LUFe on testing data')
    plt.plot(range(10,100,10), [1-item for item in mean_self_svm], label='RFE-SVM on training data')
    plt.legend(loc='best')
    plt.xlabel('percent of data')
    plt.ylabel('error')
    plt.savefig('learning curve')
    plt.show()

# collate_learning_curve(5)