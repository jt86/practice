from Get_Full_Path import get_full_path
from ExperimentSetting import Experiment_Setting
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
    output_directory = get_full_path(('Desktop/Privileged_Data/AllResults/{}/{}/{}/{}{}/').format(s.dataset,s.kernel,s.name,s.dataset, s.datasetnum))
    assert os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed))),'{} does not exist'.format(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
    with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
        for item in csv.reader(cv_lupi_file):
            results[int(item[0])]=item[1]
    # assert 0 not in results
    if 0 in results:
        # print ("setting = Experiment_Setting(foldnum={}, topk=300, dataset='tech', datasetnum={}, kernel='linear',cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv={}, percentageofinstances=100, take_top_t='top', lupimethod='{}',featsel='{}',classifier='{}')".format(np.where(results==0)[0][0],s.datasetnum,s.percent_of_priv,s.lupimethod,s.featsel,s.classifier))
        # print("print('--foldnum {} --topk {} --dataset {} --datasetnum {} --skfseed {} --lupimethod {} --featsel {} --classifier {} --stepsize {} --kernel linear  --cmin -3 --cmax 3 --numberofcs 7 --percentofpriv 100 --percentageofinstances {} --taketopt top')"
        # .format(np.where(results == 0)[0][0], 300, 'tech', s.datasetnum, 1,  s.lupimethod, s.featsel, s.classifier, s.stepsize, s.percentageofinstances))

        print('s = Experiment_Setting(foldnum={}, topk={}, dataset="{}", datasetnum={}, skfseed={}, lupimethod="{}", featsel="{}", classifier="{}", stepsize={},'
              ' kernel="linear",  cmin=-2, cmax=2, numberofcs=3, percent_of_priv=100, percentageofinstances={}) \n single_fold(s)'
            .format(np.where(results == 0)[0][0], 300, 'tech', s.datasetnum, 1, s.lupimethod, s.featsel, s.classifier,
                    s.stepsize, s.percentageofinstances))

    return results



##########################

def collate_all_datasets(s,num_datasets=295):
    all_results = []
    for datasetnum in range(num_datasets):
        s.datasetnum=datasetnum
        all_results.append(collate_single_dataset(s))
    # print(s.name,1-np.mean(all_results))
    return (np.array(all_results))



def compare_two_settings(s1, s2):
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    diffs_list = (setting_two-setting_one)
    print(s1.name,np.mean(collate_all_datasets(s1)),s2.name,np.mean(collate_all_datasets(s2)))
    print('{} : better {}; {} better: {}; equal: {}; mean improvement={}%'.format(s1.name,len(np.where(diffs_list > 0)[0]),
          s2.name,len(np.where(diffs_list < 0)[0]),len(np.where(diffs_list==0)[0]),np.mean(diffs_list)))
    print('& {} & {} & {}'.format(len(np.where(diffs_list< 0)[0]),len(np.where(diffs_list==0)[0]),len(np.where(diffs_list>0)[0])))
    return(diffs_list)



def get_graph_labels(s):
    dict_of_settings ={'svmplus':'SVM+', 'dp':r'SVM$\delta$+'}
    if s.classifier== 'baseline':
        name ='ALL-SVM'
    if s.classifier== 'lufe':
        name = 'LUFe-{}-{}'.format(s.featsel.upper(),dict_of_settings[s.lupimethod])
    if s.classifier== 'featselector':
        name = '{}-SVM'.format(s.featsel.upper())
    if s.classifier == 'lufereverse':
        name = 'LUFeReverse-{}-{}'.format(s.featsel.upper(), dict_of_settings[s.lupimethod])
    if s.classifier == 'svmreverse':
        name = '{}-SVMReverse'.format(s.featsel.upper())
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
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/{}_VS_{}'.format(s2.featsel,name1,name2)))
    plt.show()
    #

# s1 = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
#                                   take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
#
# s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                   take_top_t='top', lupimethod='nolufe', featsel='rfe', classifier='featselector')

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




def get_mi_score(labels_train,data):
    print(data.shape)
    all_scores=np.zeros(data.shape[1])
    for count,item in enumerate(np.hsplit(data, data.shape[1])):
        item=item.flatten()
        all_scores[count]=mutual_info_score(labels_train,item)
    return(all_scores)



def plot_total_comparison(s1, s2, s_baseline,num_datasets = 295):
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1)*100
    indices = np.argsort(baseline[:num_datasets])
    setting_one_scores=setting_one[indices]
    setting_two_scores = setting_two[indices]
    baseline_scores = baseline[indices]
    plt.plot(range(num_datasets),setting_one_scores[:num_datasets],color='blue',label=get_graph_labels(s1))
    plt.plot(range(num_datasets), setting_two_scores[:num_datasets],color='red',label=get_graph_labels(s2))
    plt.plot(range(num_datasets), baseline_scores[:num_datasets], color='black', label=get_graph_labels(s_baseline))
    plt.ylabel('Accuracy score (%)')
    plt.xlabel('Dataset number (sorted by accuracy score of ALL setting)')
    plt.legend(loc='best')
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/ALL_vs_{}_vs_{}'.format(s1.featsel, get_graph_labels(s1), get_graph_labels(s2))))
    plt.show()



def plot_total_comparison2(s1, s2, s_baseline,num_datasets = 295):
    '''
    Plots settings 1 and 2, sorted by how much s1 improves over ALL
    '''
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1)*100
    improvements = setting_one-baseline
    indices = np.argsort(improvements[:num_datasets])
    setting_one_errors=setting_one[indices]-baseline[indices]
    setting_two_errors = setting_two[indices]-baseline[indices]
    baseline_errors = baseline[indices]- baseline[indices]
    plt.plot(range(num_datasets),setting_one_errors[:num_datasets],color='blue',label=get_graph_labels(s1))
    plt.plot(range(num_datasets), setting_two_errors[:num_datasets],color='red',label=get_graph_labels(s2))
    plt.plot(range(num_datasets), baseline_errors[:num_datasets], color='black', label=get_graph_labels(s_baseline))
    plt.ylabel('Accuracy score (%)')
    plt.xlabel('Dataset number (sorted by improvement of {} over ALL'.format( get_graph_labels(s1),))
    plt.legend(loc='best')
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/ALL_vs_{}_vs_{}2'.format(s1.featsel, get_graph_labels(s1), get_graph_labels(s2))))
    plt.title('Improvements over ALL-SVM setting')
    plt.show()


def plot_total_comparison3(s1, s2, s_baseline,num_datasets = 295):
    '''
    Plots the improvement of s2 over s1, sorted by how much s1 improves over ALL
    '''
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1)*100
    improvements = setting_one-baseline
    indices = np.argsort(improvements[:num_datasets])

    setting_two_scores = setting_two[indices]-setting_one[indices]

    # plt.plot(range(num_datasets),setting_one_scores[:num_datasets],color='blue',label=get_graph_labels(s1))
    plt.plot(range(num_datasets), setting_two_scores[:num_datasets],color='red',label=get_graph_labels(s2))
    # plt.plot(range(num_datasets), baseline_scores[:num_datasets], color='black', label=get_graph_labels(s_baseline))
    plt.ylabel('Accuracy score (%)')
    plt.xlabel('Dataset number (sorted by improvement of {} over ALL'.format( get_graph_labels(s1),))
    plt.legend(loc='best')
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/ALL_vs_{}_vs_{}3'.format(s1.featsel, get_graph_labels(s1), get_graph_labels(s2))))
    plt.title('Improvements over ALL-SVM setting')
    plt.show()


def plot_total_comparison4(s1, s2, s_baseline,num_datasets = 295):
    '''
    Plots the improvement of s2 over s1, sorted by how much s1 improves over ALL
    '''
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1)*100
    s1_improvements_over_all = setting_one-baseline
    indices = np.argsort(s1_improvements_over_all[:num_datasets])

    s2_improvements_over_s1 = setting_two[indices]-setting_one[indices]

    # plt.plot(range(num_datasets),setting_one_scores[:num_datasets],color='blue',label=get_graph_labels(s1))
    plt.plot(s1_improvements_over_all[indices], s2_improvements_over_s1[:num_datasets],color='red',label=get_graph_labels(s2))
    # plt.plot(range(num_datasets), baseline_scores[:num_datasets], color='black', label=get_graph_labels(s_baseline))
    plt.ylabel('Accuracy score (%)')
    plt.xlabel('Dataset number (sorted by improvement of {} over ALL'.format( get_graph_labels(s1),))
    plt.legend(loc='best')
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/ALL_vs_{}_vs_{}4'.format(s1.featsel, get_graph_labels(s1), get_graph_labels(s2))))
    plt.title('Improvements over ALL-SVM setting')
    plt.show()
 
def plot_total_comparison5(s1, s2, s_baseline):
    '''
    Produces scatter plot: s1 improvement over ALL vs s2 improvement over s1
    '''
    setting_one = np.mean(collate_all_datasets(s1), axis=1)*100
    setting_two = np.mean(collate_all_datasets(s2), axis=1)*100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1)*100

    s1_improvements_over_all = setting_one-baseline
    s2_improvements_over_s1 = setting_two-setting_one
    plt.scatter(s1_improvements_over_all, s2_improvements_over_s1,color='green',label=get_graph_labels(s2))

    z = np.polyfit(s1_improvements_over_all, s2_improvements_over_s1, 1)
    p = np.poly1d(z)
    pylab.plot(s1_improvements_over_all, p(s1_improvements_over_all), "r--")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.ylabel('Improvement of {} over {}'.format(get_graph_labels(s2),get_graph_labels(s1)))
    plt.xlabel('Improvement of {} over ALL'.format( get_graph_labels(s1),))
    plt.savefig(get_full_path('Desktop/Privileged_Data/Graphs/{}/ALL_vs_{}_vs_{}5'.format(s1.featsel, get_graph_labels(s1), get_graph_labels(s2))))
    plt.show()


def plot_total_comparison6(s1, s2, s_baseline):
    '''
    Produces scatter plot: s1 improvement over ALL vs s2 improvement over ALL
    '''
    setting_one = np.mean(collate_all_datasets(s1), axis=1) * 100
    setting_two = np.mean(collate_all_datasets(s2), axis=1) * 100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1) * 100

    s1_improvements_over_all = setting_one - baseline
    s2_improvements_over_all = setting_two - baseline
    plt.scatter(s1_improvements_over_all, s2_improvements_over_all, color='green', label=get_graph_labels(s2))

    z = np.polyfit(s1_improvements_over_all, s2_improvements_over_all, 1)
    p = np.poly1d(z)
    pylab.plot(s1_improvements_over_all, p(s1_improvements_over_all), "r--")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.ylabel('Improvement of {} over ALL'.format(get_graph_labels(s2), get_graph_labels(s1)))
    plt.xlabel('Improvement of {} over ALL'.format(get_graph_labels(s1), ))
    plt.savefig(get_full_path(
        'Desktop/Privileged_Data/Graphs/{}/ALL_vs_{}_vs_{}6'.format(s1.featsel, get_graph_labels(s1),
                                                                    get_graph_labels(s2))))
    plt.show()


def plot_reverse_comparison(s1, s2, s_baseline):
    '''
    Produces scatter plot: s1 improvement over ALL vs s2 improvement over s1
    '''
    setting_one = np.mean(collate_all_datasets(s1), axis=1) * 100
    setting_two = np.mean(collate_all_datasets(s2), axis=1) * 100
    baseline = np.mean(collate_all_datasets(s_baseline), axis=1) * 100

    s1_improvements_over_all = setting_one - baseline
    s2_improvements_over_all = setting_two - baseline
    plt.scatter(s1_improvements_over_all, s2_improvements_over_all, color='green', label=get_graph_labels(s2))

    z = np.polyfit(s1_improvements_over_all, s2_improvements_over_all, 1)
    p = np.poly1d(z)
    pylab.plot(s1_improvements_over_all, p(s1_improvements_over_all), "r--")
    plt.axhline(y=0, color='k')
    plt.axvline(x=0, color='k')
    plt.ylabel('Improvement of {} over ALL'.format(get_graph_labels(s2), get_graph_labels(s1)))
    plt.xlabel('Improvement of {} over ALL'.format(get_graph_labels(s1), ))
    plt.savefig(get_full_path(
        'Desktop/Privileged_Data/Graphs/ReverseComparisons/ALL_vs_{}_vs_{}'.format(get_graph_labels(s1),
                                                                    get_graph_labels(s2))))
    plt.show()

#
# for featsel in ['anova', 'bahsic','chi2','mi','rfe']:
#
#     s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
#     s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
#     s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
#                                       take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
#
#     plot_total_comparison5(s1, s2, s_baseline)
#     plot_bars(s1,s2)
#     plot_bars(s_baseline,s2)
#     plot_bars(s_baseline,s1)

classifier = 'featselector'
lupimethod = 'nolufe'


# for featsel in ['rfe','anova','bahsic','chi2','mi']:
#
#
#     s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
#                                   take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
#
#
#     s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='linear',
#                                       take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')
#     compare_two_settings(s1,s2)



# s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1, kernel='rbf',
#                                   take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufenonlincrossval',cmin=-2,cmax=2)
# collate_single_dataset(s2)

# plot_bars(s1, s2)

# for featsel in ['rfe']:
#
#     s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='bbc', datasetnum=0, skfseed=1,
#                                       take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
#     collate_single_dataset(s_baseline)

    # s1 = Experiment_Setting(foldnum='all', topk=300, dataset='bbc', datasetnum='all', skfseed=1,
    #                                   take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
    # s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
    #                                   take_top_t='top', lupimethod='dp', featsel=featsel, classifier='lufe')