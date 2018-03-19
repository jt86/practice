from Get_Full_Path import get_full_path
from scipy.sparse import coo_matrix,csr_matrix,dok_matrix
import numpy as np
import sys
def get_bbc_data():
    with open (get_full_path('Desktop/Privileged_Data/bbc/bbc.mtx')) as bbc_file:
        bbc_data = np.array([[float(i) for i in item.strip('\n').split()] for item in (bbc_file.readlines())[2:]])
    row,column,data = (bbc_data[:,1]-1), (bbc_data[:,0]-1), (bbc_data[:,2])
    bbc_mtx=csr_matrix((data,(row,column))).todense()
    with open(get_full_path('Desktop/Privileged_Data/bbc/bbc.classes')) as bbc_file:
        bbc_labels = np.array([int(item.split()[1]) for item in bbc_file.readlines()[4:]])
    bbc_labels = np.array([1 if item in [0,1] else -1 for item in bbc_labels])
    bbc_data_pos = bbc_mtx[bbc_labels==1]
    bbc_data_neg = bbc_mtx[bbc_labels==-1]
    return(bbc_data_pos, bbc_data_neg)

bbc_data_pos, bbc_data_neg = get_bbc_data()
print(bbc_data_pos.shape,bbc_data_neg.shape)

def get_reuters(l1, l2):
    with open (get_full_path('Desktop/Privileged_Data/rcv1rcv2aminigoutte/{}/Index_{}-{}').format(l1,l2,l1))as reuters_file:
        reuters_data = np.array([item.split() for item in (reuters_file.readlines())])
        reuters_labels = np.array([item[0] for item in reuters_data])
        reuters_data = np.array([item[1:] for item in reuters_data])
        max_index = (max([int(item[-1].split(':')[0]) for item in reuters_data]))
        features_array = np.zeros(((reuters_labels.shape[0]),max_index))
        for row_num, row in enumerate(reuters_data):
            for item in row:
                index, value = int(item.split(':')[0])-1, float(item.split(':')[1])
                features_array[row_num, index] = value

    return(features_array,reuters_labels)

# lang_list = ['EN','FR','GR','SP']
# for item in lang_list:
#     for item2 in lang_list:
#         if item2!=item:
#             print(item,item2)
#             features_array,reuters_labels = (get_reuters(item,item2))
#             print(features_array.shape)
#             print(reuters_labels.shape)
#
