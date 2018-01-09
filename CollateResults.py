from Get_Full_Path import get_full_path
from SingleFoldSlice import make_directory, Experiment_Setting
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mutual_info_score
from SingleFoldSlice import get_train_and_test_this_fold, get_norm_priv

def collate_single_dataset(s):
    '''
    checks that output files exist for the input setting, and have values for all 10 folds
    if there is a missing result, print the setting that is lacking
    '''
    results=np.zeros(10)
    output_directory = get_full_path(('Desktop/Privileged_Data/AllResults/{}/{}{}/').format(s.name,s.dataset, s.datasetnum))
    assert os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed))),'{} does not exist'.format(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
    with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
        for item in csv.reader(cv_lupi_file):
            results[int(item[0])]=item[1]
    assert 0 not in results
    if 0 in results:
        # print ("setting = Experiment_Setting(foldnum={}, topk=300, dataset='tech', datasetnum={}, kernel='linear',cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv={}, percentageofinstances=100, take_top_t='top', lupimethod='{}',featsel='{}',classifier='{}')".format(np.where(results==0)[0][0],s.datasetnum,s.percent_of_priv,s.lupimethod,s.featsel,s.classifier))
        print("print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top')"
        .format(np.where(results == 0)[0][0], 300, 'tech', s.datasetnum, 1,  s.lupimethod, s.featsel, s.classifier, s.stepsize, s.percentageofinstances))
    return results




classifier = 'featselector'
lupimethod = 'nolufe'
featsel='rfe'
# for featsel in ['rfe','anova','chi2']:#w,'bahsic']:#,'mi']:#

######### For given dataset, do learning curve

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

##########################

def collate_all_datasets(s,num_datasets=295):
    all_results = []
    for datasetnum in range(num_datasets):
        s.datasetnum=datasetnum
        all_results.append(collate_single_dataset(s))
    # print(s.name,1-np.mean(all_results))
    return (np.array(all_results))
#
# classifier = 'featselector'
# lupimethod = 'nolufe'
# for featsel in ['rfe','anova','chi2']:#w,'bahsic']:#,'mi']:#
#     for instances in [10,20,30,40,50,60,70,80,90]:
#         setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', kernel='linear',
#                  cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=instances, take_top_t='top', lupimethod=lupimethod,
#                  featsel=featsel,classifier=classifier,stepsize=0.1)
#         collate_all_datasets(setting, num_datasets=10)
#
# classifier = 'lufe'
# lupimethod = 'svmplus'
# for featsel in ['rfe', 'anova', 'chi2']:  # w,'bahsic']:#,'mi']:#
#     for instances in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
#         setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', kernel='linear',
#                                      cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100,
#                                      percentageofinstances=instances, take_top_t='top', lupimethod=lupimethod,
#                                      featsel=featsel, classifier=classifier, stepsize=0.1)
#         collate_all_datasets(setting, num_datasets=10)


# dataset = 'dorothea'
# for featsel in ['rfe', 'anova', 'chi2']:#, 'mi']:
#     for lupimethod in ['dp','svmplus']:
#         s = Experiment_Setting(foldnum='all', topk=300, dataset=dataset, datasetnum='all', skfseed=1,
#                                take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufe',
#                                stepsize=0.1)
#         scores = collate_all_datasets(s, num_datasets=1)
#         print(dataset, featsel, lupimethod,np.mean(scores))


# for top in ['top','bottom']:
#     for percentofpriv in [10,25,50,75]:
#         setting = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                           take_top_t=top, lupimethod='svmplus', featsel='rfe', classifier='lufe',percent_of_priv=percentofpriv)
#         print(top,percentofpriv,np.mean(collate_all_datasets(setting)))




def compare_two_settings(s1, s2):
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    # print(setting_one)
    # print(setting_two)
    improvements_list = (setting_two-setting_one)

    for i,j,k in zip(setting_one,setting_two,improvements_list):
        print (i,j,k)
    print(improvements_list)
    print(s1.name,np.mean(collate_all_datasets(s1)),s2.name,np.mean(collate_all_datasets(s2)))
    print('{} : better {}; {} better: {}; equal: {}; mean improvement={}%'.format(s1.name,len(np.where(improvements_list > 0)[0]),
          s2.name,len(np.where(improvements_list < 0)[0]),len(np.where(improvements_list==0)[0]),np.mean(improvements_list)))
    return(improvements_list)

# for lupimethod in ['svmplus','dp','dsvm']:
#     print('\n'+lupimethod)
#     for featsel in ['rfe','anova','chi2','bahsic','mi']:
#         s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                              take_top_t='top', lupimethod=lupimethod, featsel=featsel, classifier='lufereverse',
#                       stepsize=0.1)
#         scores = collate_all_datasets(s)
#         print(featsel, np.mean(scores))
#
# print('\n svm reverse')
# for featsel in ['rfe', 'anova', 'chi2', 'bahsic', 'mi']:
#     s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                            take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='svmreverse',
#                            stepsize=0.1)
#     scores = collate_all_datasets(s)
#     print(featsel, np.mean(scores))

# old version with two names
#
# def get_graph_labels(s):
#     dict_of_settings ={'svmplus':'SVM+', 'dp':r'SVM$\delta$+'}
#     if s.classifier== 'baseline':
#         long,short = 'ALL-SVM', 'ALL-SVM'
#     if s.classifier== 'lufe':
#         long, short = '{} feature selection LUFe-SVM+'.format(s.featsel.upper()), 'LUFe-{}-{}'.format(s.featsel.upper(),dict_of_settings[s.lupimethod])
#     if s.classifier== 'featselector':
#         long,short = '{} features only SVM'.format(s.featsel.upper()), '{}-SVM'.format(s.featsel.upper())
#     if s.classifier == 'lufereverse':
#         long,short = 'Reversed LUFe','{}-LUFe-Reverse'.format(s.featsel.upper())
#     if s.classifier == 'svmreverse':
#         long, short = 'SVM trained on {} unselected features'.format(s.featsel.upper()), 'LUFe-{}'.format(s.featsel.upper())
#     return long,short

def get_graph_labels(s):
    dict_of_settings ={'svmplus':'SVM+', 'dp':r'SVM$\delta$+'}
    if s.classifier== 'baseline':
        name ='ALL-SVM'
    if s.classifier== 'lufe':
        name = 'LUFe-{}-{}'.format(s.featsel.upper(),dict_of_settings[s.lupimethod])
    if s.classifier== 'featselector':
        name = '{}-SVM'.format(s.featsel.upper())
    if s.classifier == 'lufereverse':
        name ='{}-LUFe-Reverse'.format(s.featsel.upper())
    if s.classifier == 'svmreverse':
        name = 'LUFe-{}'.format(s.featsel.upper())
    return name



def plot_bars(s1, s2):
    improvements_list = compare_two_settings(s1, s2)
    improvements_list.sort()
    name1, name2 = get_graph_labels(s1), get_graph_labels(s2)
    plt.bar(range(len(improvements_list)),improvements_list, color='black')
    # plt.title('{} VS {}\n Improvement by {} = {}%, {} of {} cases'.format(short1,short2,short1,round(np.mean(improvements_list)*100,2),len(np.where(improvements_list >= 0)[0]),len(improvements_list)))
    plt.title('{} vs {}\n Improvement by {}: mean = {}%; {} of {} cases'.format(name1,name2,name2,round(np.mean(improvements_list),2),len(np.where(improvements_list >= 0)[0]),len(improvements_list)))
    plt.ylabel('Difference in accuracy score (%)\n {} better <-----> {} better'.format(name1,name2))
    plt.xlabel('dataset index (sorted by improvement)')
    plt.ylim(-20,30)
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}_VS_{}'.format(name1,name2)))
    plt.show()
    #

s1 = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')

s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='rfe', classifier='featselector')

# s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                   take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='lufe')

# plot_bars(s1, s2)

def get_lufe_improvements_per_fold(featselector, lufe):
    lufe_improvements = collate_all_datasets(lufe)-collate_all_datasets(featselector)
    # print(featselector.featsel,'better',len(np.where(lufe_improvements > 0)[0]), 'same',(len(np.where(lufe_improvements==0)[0])),'worse',(len(np.where(lufe_improvements<0)[0])))
    return lufe_improvements

def compare_performance_with_improvement(values_for_xaxis, setting_one, setting_two, ind_folds=False):
    if type(values_for_xaxis)==Experiment_Setting:
        values_for_xaxis = collate_all_datasets(values_for_xaxis)
    else:
        values_for_xaxis = np.mean
    lufe_improvements = get_lufe_improvements_per_fold(setting_one, setting_two)
    if ind_folds==False:
        values_for_xaxis=np.mean(values_for_xaxis, axis=1)
        lufe_improvements = np.mean(lufe_improvements, axis=1)

    plt.scatter(values_for_xaxis, lufe_improvements)
    z = np.polyfit(values_for_xaxis.flatten(), lufe_improvements.flatten(), 1)
    p = np.poly1d(z)
    plt.plot(values_for_xaxis, p(values_for_xaxis), 'r')
    long,short1 = get_graph_labels(setting_one)
    long, short2 = get_graph_labels(setting_two)
    plt.xlabel('Accuracy of {}'.format(get_graph_labels(values_for_xaxis)[0]))
    plt.ylabel('Improvement by {} over {}'.format(short2,short1))#(setting_two.name.split('-')[0],setting_one.name.split('-')[0]))
    plt.title('Effect of {} unselected features'.format(setting_one.featsel.upper()))
    print ('y=%.3fx+(%.3f)'%(z[0],z[1]))
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/Investigation/delta_plus_VS_SVMplus-{}'.format(setting_one.featsel)))
    plt.show()
    # print(x_axis)



# for featsel in ['rfe','bahsic','anova','chi2','mi']:
#     print(featsel)
#     lufereverse = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='svmplus', featsel=featsel, classifier='lufereverse')
#     svmreverse = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='nolufe', featsel=featsel, classifier='svmreverse')
#     svm = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='nolufe', featsel=featsel, classifier='featselector')
#     lufe = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='svmplus', featsel=featsel,classifier='lufe')
#     svmplus = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='svmplus', featsel=featsel, classifier='lufe')
#     deltaplus = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='dp', featsel=featsel, classifier='lufe')
#     baseline = Experiment_Setting(foldnum='all', datasetnum='all', lupimethod='svmplus', featsel=featsel,classifier='lufe')
    # compare_performance_with_improvement(svmreverse, deltaplus, svmplus)

    # np.load(get_full_path('Desktop/Privileged_Data/MIScores/selected/{}/tech{}-1-{}'.format(s.featsel, s.datasetnum, s.foldnum)),get_mi_score(labels_train, normal_train))



def get_mi_score(labels_train,data):
    print(data.shape)
    all_scores=np.zeros(data.shape[1])
    for count,item in enumerate(np.hsplit(data, data.shape[1])):
        item=item.flatten()
        all_scores[count]=mutual_info_score(labels_train,item)
    return(all_scores)

#
# for datasetnum in range(295):
#     for fold in range(10):
#         s = Experiment_Setting(foldnum=fold, datasetnum=datasetnum, lupimethod='nolufe', featsel='mi', classifier='featselector')
#         all_train, all_test, labels_train, labels_test = get_train_and_test_this_fold(s)
#         normal_train, normal_test, priv_train, priv_test = get_norm_priv(s, all_train, all_test)
#
#         np.save(get_full_path('Desktop/Privileged_Data/MIScores/selected/{}/tech{}-1-{}'.format(s.featsel,s.datasetnum,s.foldnum)),get_mi_score(labels_train,normal_train))
#         np.save(get_full_path('Desktop/Privileged_Data/MIScores/unselected/{}/tech{}-1-{}'.format(s.featsel,s.datasetnum,s.foldnum)),get_mi_score(labels_train,priv_train))
#         print(get_mi_score(labels_train, priv_train))


# print(np.shape(np.load(get_full_path('Desktop/Privileged_Data/MIScores/selected/rfe/tech0-1-0.npy'))))
# print(np.shape(np.load(get_full_path('Desktop/Privileged_Data/MIScores/unselected/rfe/tech0-1-0.npy'))))


def plot_total_comparison(s1, s2, s_baseline,num_datasets = 295):
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1)*100
    indices = np.argsort(baseline[:num_datasets])
    setting_one_errors=setting_one[indices]
    setting_two_errors = setting_two[indices]
    baseline_errors = baseline[indices]
    plt.plot(range(num_datasets),setting_one_errors[:num_datasets],color='blue',label=get_graph_labels(s1))
    plt.plot(range(num_datasets), setting_two_errors[:num_datasets],color='red',label=get_graph_labels(s2))
    plt.plot(range(num_datasets), baseline_errors[:num_datasets], color='black', label=get_graph_labels(s_baseline))
    plt.ylabel('Accuracy score (%)')
    plt.xlabel('Dataset number (sorted by accuracy score of ALL setting)')
    plt.legend(loc='best')
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/ALL_vs_{}_vs_{}'.format(get_graph_labels(s1), get_graph_labels(s2))))
    plt.show()


s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')

s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='rfe', classifier='featselector')

s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel='rfe', classifier='lufe')

plot_total_comparison(s1, s2, s_baseline)