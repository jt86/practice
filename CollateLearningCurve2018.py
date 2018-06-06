from ExperimentSetting import Experiment_Setting
from CollateResults import collate_all
import numpy as np
from matplotlib import pyplot as plt
from Get_Full_Path import get_full_path
means_featsel, means_lufe, mean_self_svm, mean_self_lufe = [],[],[],[]

pc_range=range(20,100,20)

for percentageofinstances in pc_range:
    s_featsel = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                   percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                   lupimethod='nolufe', featsel='mi', classifier='featselector')

    s_lufe = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                lupimethod='svmplus', featsel='mi', classifier='lufe')

    s_feat_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
                                   percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                   lupimethod='nolufe', featsel='mi', classifier='svmtrain')

    s_lufe_train = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                kernel='linear', percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
                                lupimethod='svmplus', featsel='mi', classifier='lufetrain')

    print(np.mean(collate_all(s_featsel)), np.mean(collate_all(s_lufe)))
    means_featsel.append(np.mean(collate_all(s_featsel)))
    means_lufe.append(np.mean(collate_all(s_lufe)))
    mean_self_svm.append(np.mean(collate_all(s_feat_train)))
    mean_self_lufe.append(np.mean(collate_all(s_lufe_train)))

plt.plot(pc_range,[100-item for item in means_featsel], label='MI-SVM on testing data')
plt.plot(pc_range,[100-item for item in means_lufe], label='MI-LUFe on testing data')
plt.plot(pc_range,[100-item for item in mean_self_svm], label='MI-SVM on training data')
plt.plot(pc_range,[100-item for item in mean_self_lufe], label='MI-LUFe on training data')

plt.legend(loc='best')
plt.xlabel('Percent of training data used')
plt.ylabel('Mean error (%)')
plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/chap3b/learningcurve.pdf'),format='pdf')
plt.show()