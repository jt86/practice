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


# plot_total_comparison(s1,s2,s_baseline)
# plot_bars(s_baseline,s1)
# plot_bars(s_baseline,s2)
# plot_bars(s1,s2)


############ Top 500

s3 = Experiment_Setting(foldnum='all', topk=500, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='nolufe', featsel=featsel, classifier='featselector')
s4 = Experiment_Setting(foldnum='all', topk=500, dataset='tech', datasetnum='all', skfseed=1,
                                  take_top_t='top', lupimethod='svmplus', featsel=featsel, classifier='lufe')

plot_total_comparison(s3,s4,s_baseline)
# plot_bars(s_baseline,s3)
# plot_bars(s_baseline,s4)
# plot_bars(s3,s4)

# ############ For MTL comparison

# featsel='rfe'
#
# mtl_results = ((collate_mtl_results(featsel.upper(), 300)))
# lufe_rbf_setting = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=all, kernel='rbf', skfseed=1,
#                                   percentageofinstances=100, take_top_t='top', lupimethod='svmplus',
#                                   featsel=featsel, classifier='lufe')
# all_results=dict()
# all_results['LUFe-RFE-SVM+'] = collate_all(s2)
# all_results['LUFe-RFE-RBF-SVM+'] = collate_all(lufe_rbf_setting)
# all_results['LUFe-MTL'] = mtl_results
# plot_bars_for_mtl(all_results['LUFe-MTL'] , all_results['LUFe-RFE-SVM+'], 'LUFe-MTL', 'LUFe-RFE-SVM+', featsel)
# plot_bars_for_mtl(all_results['LUFe-MTL'] , all_results['LUFe-RFE-RBF-SVM+'], 'LUFe-MTL', 'LUFe-RFE-RBF-SVM+', featsel)
# plot_bars_for_mtl(all_results['LUFe-RFE-RBF-SVM+'] , all_results['LUFe-RFE-SVM+'], 'LUFe-RFE-RBF-SVM+', 'LUFe-RFE-SVM+', featsel)
