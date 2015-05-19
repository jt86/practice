__author__ = 'jt306'
import os
from Get_Full_Path import get_full_path

list_of_t=[]
peeking = False
num_folds=10
rank_metric='r2'
cmin=0
cmax=4
cstarmin,cstarmax=None,None
number_of_cs=5


for awanum in range(10):
    dataset='awa{}'.format(awanum)

    output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/FixedCandCStar13/'),dataset)
    cross_validation_folder = os.path.join(output_directory,'cross-validation')

    with open(os.path.join(cross_validation_folder,'keyword.txt'),'a') as keyword_file:
        keyword_file.write("{} t values:{}\n peeking={}; {} folds; metric: {}; c={{10^{}..10^{}}}; c*={{10^{}..10^{}}} ({} values)".format(dataset,
                list_of_t, peeking, num_folds, rank_metric, cmin, cmax, cstarmin, cstarmax, number_of_cs))
