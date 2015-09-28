__author__ = 'jt306'
import numpy as np
import itertools
from scipy import stats

list_of_lists = [[5,5,5],[43,3],[3,1,5,6,1]]

def get_mean_from(list_of_lists):
    results = []
    izipped = list(itertools.zip_longest(*list_of_lists, fillvalue=np.nan))
    # print 'getting mean'
    # print 'izipped',izipped
    for int in range(len(izipped)):
        results.append(np.nanmean(izipped[int]))
    return results

def get_error_from(list_of_lists):
    errors = []
    izipped = list(itertools.zip_longest(*list_of_lists, fillvalue=np.nan))
    # print 'getting error'
    # print 'izipped',izipped
    for int in range(len(izipped)):
        errors.append(stats.sem(izipped[int]))
    return errors
