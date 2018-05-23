from ExperimentSetting import Experiment_Setting
from CollateResults import plot_total_comparison, plot_bars
from CollateMTLResults2 import collate_mtl_results, collate_all, plot_bars_for_mtl
# from CollateResults2 import

############ FOR CHAPTER 1

featsel='rfe'
############ Top 300

s_baseline = Experiment_Setting(foldnum='all', topk='all', dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel='nofeatsel', classifier='baseline')
s1 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
s2 = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')


plot_total_comparison(s1,s2,s_baseline)
plot_bars(s_baseline,s1)
plot_bars(s_baseline,s2)
plot_bars(s1,s2)


############ Top 500

s3 = Experiment_Setting(foldnum='all', topk=500, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
s4 = Experiment_Setting(foldnum='all', topk=500, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')

plot_total_comparison(s3,s4,s_baseline)
plot_bars(s_baseline,s3)
plot_bars(s_baseline,s4)
plot_bars(s3,s4)

############ For MTL comparison

featsel='rfe'
for kernel in ['linear']:
    # for featsel in ['rfe', 'anova', 'bahsic', 'chi2', 'mi', 'rfe']:
        mtl_results = ((collate_mtl_results(featsel.upper(), 300)))

        lufe_setting = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel=kernel,
                                          cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100,
                                          percentageofinstances=100,
                                          take_top_t='top', lupimethod='svmplus',
                                          featsel=featsel, classifier='lufe', stepsize=0.1)

        svm_setting = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel=kernel,
                                         cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100,
                                         percentageofinstances=100,
                                         take_top_t='top', lupimethod='nolufe',
                                         featsel=featsel, classifier='featselector', stepsize=0.1)

        lufe_results = collate_all(lufe_setting)
        svm_results = collate_all(svm_setting)
        plot_bars_for_mtl(mtl_results, svm_results, featsel, kernel, 'featselector')

        # compare_lufe_mtl(featsel, lufe_setting, kernel)