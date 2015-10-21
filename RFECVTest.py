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
# print (y)
positiveins = X[y==1]
negativeins = X[y==0]
np.savetxt(get_full_path('Desktop/Privileged_Data/syn_positive.txt'),positiveins)
np.savetxt(get_full_path('Desktop/Privileged_Data/syn_negative.txt'),negativeins)

for i in range (2,15):
    print ('\n')
    num_folds = i


    svc = SVC(kernel="linear", C=1)
    rfecv = RFECV(estimator=svc, step=1, cv=StratifiedKFold(y, num_folds),
                  scoring='accuracy')#,verbose=10)
    rfecv.fit(X, y)

    print("Optimal number of features : %d" % rfecv.n_features_)


    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    # plt.show()


    grid_scores = rfecv.grid_scores_
    # print ('grid scores',grid_scores)
    print ('best rfecv score',max(grid_scores))
    print ('...achieved w this many feats:', np.argmax(grid_scores)+1)
    # print (grid_scores[9])

    ############################
    svc = SVC(kernel="linear", C=1)
    new_rfe = RFE(estimator=svc, step=1, n_features_to_select=np.argmax(grid_scores)+1)
    new_rfe.fit(X,y)
    # print ('new rfe score',new_rfe.score(X,y))

    list_of_scores = []
    skf = StratifiedKFold(y,n_folds=num_folds)
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        new_rfe.fit(X_train,y_train)
        ACC = new_rfe.score(X_test, y_test)
        list_of_scores.append(ACC)

    print ('normal rfe score',np.mean(list_of_scores))




# for index,(i,j) in enumerate(zip(rfecv.support_, new_rfe.support_)):
#     if i!=j:
#         print (index,' do not match')

# print ('rfecv \n', [pair for pair in enumerate(rfecv.ranking_)])
# print ('rfe \n', [pair for pair in enumerate(new_rfe.ranking_)])
# print ('rfecv \n', rfecv.support_)
# print ('rfe \n', new_rfe.support_)
# print('rfecv est', rfecv.estimator_)
# print ('rfe est',new_rfe.estimator_)
# print (np.argwhere(rfecv.support_))
# print (np.argwhere(new_rfe.support_))



# list_of_rfecv_scores = []
# skf1 = StratifiedKFold(y,n_folds=10)
# for train_index, test_index in skf1:
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#     svc.fit(X_train,y_train)
#     ACC = svc.score(X_test, y_test)
#     list_of_rfecv_scores.append(ACC)
#
# print ('rfecv',np.mean(list_of_rfecv_scores))
# print ('rfecv',list_of_rfecv_scores)
#
# print ('rfe',np.mean(list_of_scores))
# print ('rfe',list_of_scores)
