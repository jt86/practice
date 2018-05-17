'''
Used to get info about dimensions of datasets to write about
'''

from GetSingleFoldData import load_dataset_from_name
from Get_Full_Path import get_full_path
import numpy as np

# Go through datasets and save the dimensions of each
all_dims = np.zeros((295,2))
for index in range(295):
    i,j = (load_dataset_from_name('tech',index))
    all_dims[index,0]=i.shape[0]+j.shape[0]
    all_dims[index, 1] = i.shape[1]
print(all_dims)
np.save(get_full_path('Desktop/Privileged_Data/all_tech_dimensions'),all_dims)

# Print the max and min of
all_dims = np.load(get_full_path('Desktop/Privileged_Data/all_tech_dimensions.npy'))
print (np.max(all_dims,axis=0))
print (np.min(all_dims,axis=0))
print (np.mean(all_dims,axis=0))
print (np.median(all_dims,axis=0))