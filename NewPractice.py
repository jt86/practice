import numpy as np
from Get_Full_Path import get_full_path
data = np.load(get_full_path('Desktop/Privileged_Data/SavedIndices/top{}RFE/{}{}-{}-{}.npy'.format(300,'tech',245,4,3)))
print(data.shape)