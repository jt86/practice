__author__ = 'jt306'
from BringFoldsTogether import read_from_disk_and_plot

num_folds = 10
# x_axis_list = range (1,13)
x_axis_list = [5,10,25,50,75]
list_of_values = x_axis_list


for dataset_num in range(1):
    cross_validation_folder = '/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/FixedCandCStar3/awa{}/cross-validation'.format(dataset_num)
    read_from_disk_and_plot(num_folds, cross_validation_folder, list_of_values, x_axis_list)