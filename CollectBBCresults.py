from ExperimentSetting import Experiment_Setting


for foldnum in range(10):
    s = Experiment_Setting(foldnum=foldnum, topk='all', dataset='bbc', datasetnum=0, kernel='linear',
                                   cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                                   take_top_t='top', lupimethod='nolufe',
                                   featsel='nofeatsel', classifier='baseline', stepsize=0.1)


    s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='bbc', datasetnum=0, kernel='linear',
                                   cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                                   take_top_t='top', lupimethod='nolufe',
                                   featsel='rfe', classifier='featselector', stepsize=0.1)


    s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='bbc', datasetnum=0, kernel='linear',
                                   cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                                   take_top_t='top', lupimethod='svmplus',
                                   featsel='rfe', classifier='lufe', stepsize=0.1)
