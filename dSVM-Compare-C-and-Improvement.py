from collections import defaultdict
import os
import numpy as np
from scipy import stats
root='/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
from Get_Full_Path import get_full_path
c_values=[]
c_star_values=[]
import matplotlib.pyplot as plt


############## FIRST - save C Values

#
# for datasetnum in range (295):
#     list = []
#     for seed in range(10):
#         experiment='/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top300chosen-100percentinstances/cross-validation{}'.format(datasetnum,seed)
#         location = os.path.join(root,experiment,'C_fullset-crossvalid-{}-300.txt'.format(datasetnum))
#         with open(location) as filename:
#             lines=filename.readlines()
#             for index,line in enumerate(lines):
#                 # print (index)
#                 if index != 0:
#                     output=line.split(' ')[-1].strip('\n')
#                     list.append(output)
#     c_values.append(list)
#
# print(len(c_values))
# c_values=np.array(c_values)
#
# np.save(get_full_path('Desktop/priviliged_c_values'),c_values)
#
# print (c_values.shape)


############  load cvalues (saved above) ... save counts for all datasets


# c_values = np.load(get_full_path('Desktop/priviliged_c_values.npy'))
#
# print(improvements.shape,c_values.shape)
#
# counts_for_all_datasets = []
# values = np.array([0.001,0.01,0.1,1.,10.,100.,1000.])
# for single_dataset in c_values:
#     counts_for_single_dataset = np.zeros(7)
#     for item in single_dataset:
#         # print(item)
#         index_of_c =(np.where(values==float(item))[0][0])
#         counts_for_single_dataset[index_of_c]+=1
#     print(counts_for_single_dataset)
#     counts_for_all_datasets.append(counts_for_single_dataset)
#
# counts_for_all_datasets=np.array(counts_for_all_datasets)
# print(counts_for_all_datasets.shape)
#
# np.save(get_full_path('Desktop/counts_for_all_datasets'),counts_for_all_datasets)



########### load counts for all datasets

counts_for_all_datasets = np.load(get_full_path('Desktop/counts_for_all_datasets.npy'))

print(counts_for_all_datasets.shape)


#NOW load improvements list (saved in SaveDSVMResults.py)

improvements = np.load(get_full_path('Desktop/dSVMimprovementsVsRFE.npy'))
print(np.argsort(improvements))

counts_for_all_datasets=counts_for_all_datasets[np.argsort(improvements)]
counts_for_all_datasets=counts_for_all_datasets[200:295,:]


ind = np.arange(200,295)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence


p1 = plt.bar(ind, counts_for_all_datasets[:,0],  color='black',edgecolor = "none", label = 0.001)
p2 = plt.bar(ind, counts_for_all_datasets[:,1],  color='black', bottom=counts_for_all_datasets[:,0], edgecolor = "none",label = 0.01)
p3 = plt.bar(ind, counts_for_all_datasets[:,2],  color='black', bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1],edgecolor='none',label = 0.1)
p4 = plt.bar(ind, counts_for_all_datasets[:,3],  color='white', bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2],edgecolor='none',label = 1.)
p5 = plt.bar(ind, counts_for_all_datasets[:,4],  color='white', bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2]+counts_for_all_datasets[:,3],edgecolor='none',label = 10.)
p6 = plt.bar(ind, counts_for_all_datasets[:,5], color='white',bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2]+counts_for_all_datasets[:,3]+counts_for_all_datasets[:,4],edgecolor='none',label = 100.)
p7  =plt.bar(ind, counts_for_all_datasets[:,6], color='white',bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2]+counts_for_all_datasets[:,3]+counts_for_all_datasets[:,4]+counts_for_all_datasets[:,5],edgecolor='none',label = 1000.)


# p1 = plt.bar(ind, counts_for_all_datasets[:,0],  color='r',edgecolor = "none", label = 0.001)
# p2 = plt.bar(ind, counts_for_all_datasets[:,1],  color='orange', bottom=counts_for_all_datasets[:,0], edgecolor = "none",label = 0.01)
# p3 = plt.bar(ind, counts_for_all_datasets[:,2],  color='pink', bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1],edgecolor='none',label = 0.1)
# p4 = plt.bar(ind, counts_for_all_datasets[:,3],  color='purple', bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2],edgecolor='none',label = 1.)
# p5 = plt.bar(ind, counts_for_all_datasets[:,4],  color='blue', bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2]+counts_for_all_datasets[:,3],edgecolor='none',label = 10.)
# p6 = plt.bar(ind, counts_for_all_datasets[:,5], color='grey',bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2]+counts_for_all_datasets[:,3]+counts_for_all_datasets[:,4],edgecolor='none',label = 100.)
# p7  =plt.bar(ind, counts_for_all_datasets[:,6], color='black',bottom=counts_for_all_datasets[:,0]+counts_for_all_datasets[:,1]+counts_for_all_datasets[:,2]+counts_for_all_datasets[:,3]+counts_for_all_datasets[:,4]+counts_for_all_datasets[:,5],edgecolor='none',label = 1000.)

#
# plt.ylabel('Scores')
# plt.title('Scores by group and gender')
# plt.xticks(ind + width/2., ('G1', 'G2', 'G3', 'G4', 'G5'))
# plt.yticks(np.arange(0, 81, 10))
# plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0]))


plt.legend(loc="upper left", bbox_to_anchor=(1,1))#, p3[0], p4[0], p5[0], p6[0], p7[0]))

print(improvements[163])
print(improvements[226])
plt.xlabel('ranking of dSVM+ vs RFE improvement ( dsvm worse <--- ----> dsvm better)')
plt.ylabel('number of times each C value chosen')
plt.show()

# print('c counts \n',c_counts)
#
# list_of_pairs1 =[]
# for c_value,count in c_counts.items():
#     list_of_pairs1.append([c_value, count])
# list_of_pairs1.sort()
#
# for item in list_of_pairs1:
#     print('C = {} selected {} times '.format(item[0],item[1]))
# print('\n')

