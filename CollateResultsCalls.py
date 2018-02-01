# this file is a dumping ground for previous calls to CollateResults.py


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
