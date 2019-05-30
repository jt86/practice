from ExperimentSetting import Experiment_Setting
from SingleFoldSlice import single_fold
import time

datasetnum=0
i=0
time0=time.time()


i=0
#
# for featsel in ['mi','rfe','anova','bahsic','chi2']:
#     for datasetnum in range(295):
#         for foldnum in range(8,10):
#             s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
#                                     percent_of_priv=100, percentageofinstances=100, take_top_t='top',
#                                    lupimethod='svmplus', featsel=featsel, classifier='lufeauto')
#             single_fold(s)
#             i += 1
#             print('\n iteration {}; time = {}'.format(i, time.time() - time0))
# featsel='mi'
# for datasetnum in range(10):
#     s = Experiment_Setting(foldnum=0, topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
#                             percent_of_priv=100, percentageofinstances=100, take_top_t='top',
#                            lupimethod='nolufe', featsel=featsel, classifier='featselector')
#     single_fold(s)
#     print(time.time()-time0)

# i=0
# for percentageofinstances in range(20,101,20):
#     print(percentageofinstances)
#     print(percentageofinstances)
#     for datasetnum in range(295):
#         for foldnum in range(6,8):
#             s = Experiment_Setting(foldnum=foldnum, topk='nofeatsel', dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
#                                     percent_of_priv=100, percentageofinstances=percentageofinstances, take_top_t='top',
#                                    lupimethod='nolufe', featsel='nofeatsel', classifier='baselinetrain')
#             single_fold(s)
#             i += 1
#             print('\n iteration {}; time = {}'.format(i, time.time() - time0))

# s = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=9, kernel='linear',
#                        cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=75, percentageofinstances=100,
#                        take_top_t='bottom', lupimethod='svmplus',
#                        featsel='mi', classifier='lufe', stepsize=0.1)
# single_fold(s)
# s = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=108, kernel='linear',
#                        cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=75, percentageofinstances=100,
#                        take_top_t='bottom', lupimethod='svmplus',
#                        featsel='mi', classifier='lufe', stepsize=0.1)
# single_fold(s)
# s = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=210, kernel='linear',
#                        cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=75, percentageofinstances=100,
#                        take_top_t='bottom', lupimethod='svmplus',
#                        featsel='mi', classifier='lufe', stepsize=0.1)
#
# single_fold(s)
# s = Experiment_Setting(foldnum=8, topk=300, dataset='tech', datasetnum=210, kernel='linear',
#                        cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=75, percentageofinstances=100,
#                        take_top_t='bottom', lupimethod='svmplus',
#                        featsel='mi', classifier='lufe', stepsize=0.1)
# single_fold(s)

# TEST
# i=0
# time0=time.time()
# percent_of_priv = 50
#
# for datasetnum in range(291,295):
#     for foldnum in range(10):
#         s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
#                                cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=percent_of_priv, percentageofinstances=100,
#                                take_top_t='bottom', lupimethod='svmplus',
#                                featsel='mi', classifier='lufe', stepsize=0.1)
#
#         single_fold(s)
#         i+=1
#         print('\n iteration {}; time = {}'.format(i, time.time()-time0))

# TEST
# i=0
# time0=time.time()
# for percent_of_priv in [10,25,50,75]:
#     for foldnum in range(10):
#         for datasetnum in range(50):
#             s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
#                                    cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=percent_of_priv, percentageofinstances=100,
#                                    take_top_t='bottom', lupimethod='svmplus',
#                                    featsel='mi', classifier='lufe', stepsize=0.1)
#
#             single_fold(s)
#             i+=1
#             print('\n iteration {}; time = {}'.format(i, time.time()-time0))

#
# time0=time.time()
# for foldnum in range(10):
#
#     s = Experiment_Setting(foldnum=foldnum, topk=500, dataset='tech', datasetnum=0, kernel='linear',
#                                    cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                                    take_top_t='top', lupimethod='nolufe',
#                                    featsel='rfe', classifier='featselector', stepsize=0.1)
#     single_fold(s)
#     print(time.time()-time0)
#     s = Experiment_Setting(foldnum=foldnum, topk=500, dataset='tech', datasetnum=0, kernel='linear',
#                                    cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                                    take_top_t='top', lupimethod='svmplus',
#                                    featsel='rfe', classifier='lufe', stepsize=0.1)
#     single_fold(s)
#     print(time.time()-time0)

# for datasetnum in range(295):
#     for foldnum in range(4):
#         s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='rbf',
#                                        cmin=-2, cmax=2, numberofcs=3, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                                        take_top_t='top', lupimethod='svmplus',
#                                        featsel='rfe', classifier='lufenonlincrossval', stepsize=0.1)
#         single_fold(s)
#
#


# featsel='rfe'
# for datasetnum in range(250,295):
#     for topk in range(320,500,20):
#         for foldnum in range(10):
#             s = Experiment_Setting(foldnum=foldnum, topk=topk, dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
#                                     percent_of_priv=100, percentageofinstances=100, take_top_t='top',
#                                    lupimethod='nolufe', featsel=featsel, classifier='featselector')
#             single_fold(s)
#             i += 1
#             print('\n iteration {}; time = {}'.format(i, time.time() - time0))




featsel='rfe'
for datasetnum in range(275,295):
    for foldnum in range(10):
        s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
                                percent_of_priv=100, percentageofinstances=100, take_top_t='top',
                               lupimethod='nolufe', featsel='random', classifier='random_featsel_svm')
        single_fold(s)

        s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
                                percent_of_priv=100, percentageofinstances=100, take_top_t='top',
                               lupimethod='svmplus', featsel='random', classifier='random_featsel_svmplus')
        single_fold(s)