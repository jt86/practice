import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV, RFE
from sklearn.datasets import make_classification
import pdb
import numpy as np
from Get_Full_Path import get_full_path
# Build a classification task using 3 informative features
X, y = make_classification(n_samples=1000, n_features=25, n_informative=3,
                           n_redundant=2, n_repeated=0, n_classes=2,
                           n_clusters_per_class=1, random_state=0)
print (y)
positiveins = X[y==1]
negativeins = X[y==0]
np.savetxt(get_full_path('Desktop/Privileged_Data/syn_positive.txt'),positiveins)
np.savetxt(get_full_path('Desktop/Privileged_Data/syn_negative.txt'),negativeins)


# Create the RFE object and compute a cross-validated score.
svc = SVC(kernel="linear", C=1)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, 2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()


new_rfe = RFE(estimator=svc, step=1, n_features_to_select=10)
new_rfe.fit(X,y)
print (new_rfe.score(X,y))

list_of_scores = []
skf = StratifiedKFold(y,n_folds=10)
for train_index, test_index in skf:
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    new_rfe.fit(X_train,y_train)
    ACC = new_rfe.score(X_test, y_test)
    list_of_scores.append(ACC)

print (np.mean(list_of_scores))