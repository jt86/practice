from collections import defaultdict
import os
import numpy as np
root='/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances'
from Get_Full_Path import get_full_path
c_values=[]
c_star_values=[]


for datasetnum in range (295):
    for seed in range(10):
        experiment='/Volumes/LocalDataHD/j/jt/jt306/Desktop/dSVM295-10x10-tech-ALLCV-3to3-featsscaled-step0.1-100toppercentpriv-100percentinstances/tech{}/top300chosen-100percentinstances/cross-validation{}'.format(datasetnum,seed)
        location = os.path.join(root,experiment,'C_fullset-crossvalid-{}-300.txt'.format(datasetnum))
        with open(location) as filename:
            lines=filename.readlines()
            for index,line in enumerate(lines):
                if index != 0:
                    output=line.split(' ')[-1].strip('\n')
                    c_values.append(output)

print(len(c_values))
c_values=np.array(c_values)

np.save(os.path.join(root,'priviliged_c_values'),c_values)


c_values= np.load(os.path.join(root,'priviliged_c_values.npy'))
c_counts=defaultdict(int)


for item in c_values:
    c_counts[item]+=1

# print('c counts \n',c_counts)

list_of_pairs1 =[]
for c_value,count in c_counts.items():
    list_of_pairs1.append([c_value, count])
list_of_pairs1.sort()

for item in list_of_pairs1:
    print('C = {} selected {} times '.format(item[0],item[1]))
print('\n')


