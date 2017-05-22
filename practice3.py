import numpy as np
from Get_Full_Path import get_full_path

array1 = np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top300mi/tech0-0-0.npy'))
array2= np.load(get_full_path('Desktop/Privileged_Data/SavedNormPrivIndices/top300mi/tech0-0-1.npy'))
print(array1[:10])
print(array2[:10])