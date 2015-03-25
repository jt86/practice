__author__ = 'jt306'
import os, sys
from Get_Full_Path import get_full_path

import os, sys
import numpy as np
import matplotlib.pyplot as plt
import logging

rootdir = get_full_path('privileged_data/results/')

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if 'chosen' in os.path.join(subdir, file):
            path = os.path.join(subdir, file)
            # SVM_dict = {}
            # SVM_plus_dict = {}
            SVM_list = []
            SVM_plus_list = []
            for line in open(path, 'r'):
                line = line.split(',')
                if len(line) >= 2:
                    gamma = line[4]
                    # logging.info( gamma)
                    if line[2] == 'SVM+':
                        SVM_plus_list += [gamma]
                    elif line[2] == 'SVM':
                        SVM_list += [gamma]

print(SVM_plus_list)
print(SVM_list)

print(SVM_plus_list)
print(len(SVM_list)

SVM_plus_list = sorted([float(item) for item in SVM_plus_list])[:-20]
SVM_list = [float(item) for item in SVM_plus_list][:-20]

hist, bins = np.histogram(SVM_plus_list, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title("Distribution of gamma for SVM+")
plt.show()
#
# plt.axis([0.0,600.0, 10000.0,20000.0])
# ax = p.gca()
# ax.set_autoscale_on(False)


hist, bins = np.histogram(SVM_list, bins=50)
width = 0.7 * (bins[1] - bins[0])
center = (bins[:-1] + bins[1:]) / 2
plt.bar(center, hist, align='center', width=width)
plt.title("Distribution of gamma for normal SVM")
plt.show()

