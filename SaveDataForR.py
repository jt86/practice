from GetSingleFoldData import get_train_and_test_this_fold
import numpy as np

dataset='tech'; datasetnum=1; k=1; skf_seed=1

train_data, test_data, train_labels, test_labels = get_train_and_test_this_fold(dataset,datasetnum,k, skf_seed)

np.save('/home/j/jt/jt306/Documents/CVPR2016_Rcode/{}-{}-{}-{}-train_data'.format(dataset,datasetnum,k,skf_seed),train_data)


practice_array = np.array([[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]])
print (practice_array.shape)
np.save('/home/j/jt/jt306/Documents/CVPR2016_Rcode/practice_array',practice_array)