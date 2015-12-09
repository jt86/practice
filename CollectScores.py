__author__ = 'jt306'
import os
from Get_Full_Path import get_full_path

dataset='tech'
cmin=-3
cmax=3
stepsize=0.1
topk=300

all_weights = []
for datasetnum in range(49):
    output_directory = get_full_path(('Desktop/Privileged_Data/GetScore-{}{}-{}to{}-{}-{}-tech{}').format(dataset,datasetnum,cmin,cmax,stepsize,topk,datasetnum))
    for k in range(10):
        for skfseed in range(10):
            with open(os.path.join(output_directory,'normal-all-f_classif-{}-{}.csv'.format(k,skfseed)),'a') as normal_chi2_file:
                all_weights+=normal_chi2_file.readline()

print (all_weights.shape)
