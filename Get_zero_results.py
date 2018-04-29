from Get_Full_Path import get_full_path
from ExperimentSetting import Experiment_Setting
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import pylab
from sklearn.metrics import mutual_info_score
from SingleFoldSlice import get_train_and_test_this_fold, get_norm_priv
import seaborn

def get_zero_results(s):
    '''
    checks that output files exist for the input setting, and have values for all 10 folds
    if there is a missing result, print the setting that is lacking
    '''
    results=np.zeros(10)
    # print('kernel',s.kernel)
    output_directory = get_full_path(('Desktop/Privileged_Data/AllResults/{}/{}/{}/{}{}/').format(s.dataset,s.kernel,s.name,s.dataset, s.datasetnum))
    if os.path.exists(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed))):#,'{} does not exist'.format(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)))
        with open(os.path.join(output_directory, '{}-{}.csv'.format(s.classifier, s.skfseed)), 'r') as cv_lupi_file:
            for item in csv.reader(cv_lupi_file):
                results[int(item[0])]=item[1]
    # assert 0 not in results
        if 0 in results:
            # print('s = Experiment_Setting(foldnum={}, topk={}, dataset="{}", datasetnum={}, skfseed={}, lupimethod="{}", featsel="{}", classifier="{}", stepsize={},'
            #       ' kernel="linear",  cmin=-3, cmax=3, numberofcs=7, percent_of_priv=100, percentageofinstances={})'
            #     .format(np.where(results == 0)[0][0], 300, 'tech', s.datasetnum, 1, s.lupimethod, s.featsel, s.classifier,
            #             s.stepsize, s.percentageofinstances))
            # print(results, np.where(results == 0)[0], datasetnum)
            print(np.where(results == 0)[0],datasetnum)
    else:
        print (s.datasetnum)

for datasetnum in range(295):
    s = Experiment_Setting(foldnum='all', topk=300, dataset='tech', datasetnum=datasetnum, skfseed=1, kernel='linear',
                                      take_top_t='bottom', lupimethod='svmplus', featsel='mi', classifier='lufe',
                            percent_of_priv=75)
    (get_zero_results(s))