__author__ = 'jt306'
import matplotlib as plt
from Get_Full_Path import get_full_path
import os
from matplotlib import pyplot as plt
import numpy as np

list_of_values = [300,500]


x = list(range(49))
y = list(range(49))


list_of_baselines=[]
list_of_300_rfe=[]

for i in range(49):
    print ('doing dataset',i)
    output_directory = (get_full_path('Desktop/Privileged_Data/TestBaseline/'))

    with open(os.path.join(output_directory,'baseline{}.csv'.format(i)),'r') as baseline_file:
        baseline_i_list = baseline_file.readline().split(',')[:-1]
        baseline_i_list = list(map(float, baseline_i_list))
        list_of_baselines.append(baseline_i_list)




print (len(np.mean(list_of_baselines,axis=1)))
list_of_errors = list([1-mean for mean in np.mean(list_of_baselines,axis=1)])

plt.plot(list(range(49)),list_of_errors)
plt.show()