__author__ = 'jt306'
import os
from Get_Full_Path import get_full_path

dataset='awa0'
output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/FixedCandCStar5/'),dataset)
fold_num=0
top_k=5


def collect_best_rfe_param(fold_num, top_k_percent, output_directory):
    with open (os.path.join(output_directory,'{}-{}.txt'.format(fold_num,top_k_percent)),'r') as best_rfe_param_file:
        return int(float(best_rfe_param_file.readline()))

print collect_best_rfe_param(0,5, output_directory)
