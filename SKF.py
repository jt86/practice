__author__ = 'jt306'
from Heart import get_heart_data
from sklearn.svm import SVC
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.feature_selection import RFECV, RFE

features_array, labels_array = get_heart_data()

cv=StratifiedKFold(labels_array, 5, shuffle=True , random_state=1)

for train, test in cv:
    print "\n",test