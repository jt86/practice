from collections import defaultdict
import os
import numpy as np
root='/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
from Get_Full_Path import get_full_path
c_values=[]
c_star_values=[]



############## FIRST - save C Values

# for datasetnum in range (295):
#     for seed in range(10):
#         experiment='/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top300chosen-100percentinstances/cross-validation{}'.format(datasetnum,seed)
#         location = os.path.join(root,experiment,'Cstar-crossvalid-{}-300.txt'.format(datasetnum))
#         with open(location) as filename:
#             lines=filename.readlines()
#             for line in lines:
#                 if '=>' in line:
#                     output=(line.split(' ')[-1])
#                     results = (float(output.split(',')[0].split('=')[1]),float(output.split(',')[1].split('=')[1]))
#                     # print (results)
#                     c_values.append(results[0])
#                     c_star_values.append(results[1])
#
# print(len(c_values),len(c_star_values))
#
# c_values=np.array(c_values)
# c_star_values=np.array(c_star_values)
#
# np.save(os.path.join(root,'c_values'),c_values)
# np.save(os.path.join(root,'c_star_values'),c_star_values)



############# THEN load C values

c_values= np.load(os.path.join(root,'c_values.npy'))
c_star_values = np.load(os.path.join(root,'c_star_values.npy'))
print(c_values.shape)
print(c_star_values.shape)

c_counts=defaultdict(int)
c_star_counts=defaultdict(int)

output_directory = get_full_path('Desktop/Privileged_Data/dSVM295-CHECK-VALUE')

for item in c_values:
    c_counts[item]+=1
for item in c_star_values:
    c_star_counts[item]+=1

print('c counts \n',c_counts)
print('c star counts \n',c_star_counts)

list_of_pairs1 =[]
for c_value,count in c_counts.items():
    list_of_pairs1.append([c_value, count])
list_of_pairs1.sort()

for item in list_of_pairs1:
    print('C = {} selected {} times '.format(item[0],item[1]))
print('\n')
list_of_pairs2 =[]
for c_value,count in c_star_counts.items():
    list_of_pairs2.append([c_value, count])
list_of_pairs2.sort()

for item in list_of_pairs2:
    print('C* = {} selected {} times '.format(item[0],item[1]))

