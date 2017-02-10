import numpy as np
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn import preprocessing
privileged_features_training = np.array([[1,3],[2,4],[5,6],[-1,-4],[-6, -5],[-1,-2]])
training_labels  = np.array([1,1,1,-1,-1,-1])
plt.scatter(privileged_features_training[:3,0],privileged_features_training[:3,1],color='blue')
plt.scatter(privileged_features_training[3:,0],privileged_features_training[3:,1],color='red')
# plt.show()

svc = SVC(C=1, kernel='linear')
svc.fit(privileged_features_training,training_labels)
print(svc.decision_function(privileged_features_training))
print('\n\n')
d_i = np.array([1 - (training_labels[i] * svc.decision_function(privileged_features_training)[i]) for i in
                range(len(training_labels))])
d_i = preprocessing.scale(d_i)
d_i = np.reshape(d_i, (d_i.shape[0], 1))
print(d_i)
print (np.mean(d_i))
print (np.std(d_i))