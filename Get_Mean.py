__author__ = 'jt306'
import numpy as np
import itertools
from scipy import stats

list_of_lists = [[5,5,5],[43,3],[3,1,5,6,1]]

def get_mean_from(list_of_lists):
    results = []
    izipped = list(itertools.izip_longest(*list_of_lists, fillvalue=np.nan))
    print 'getting mean'
    print 'izipped',izipped
    for int in range(len(izipped)):
        results.append(np.nanmean(izipped[int]))
    return results

def get_error_from(list_of_lists):
    results = []
    izipped = list(itertools.izip_longest(*list_of_lists, fillvalue=np.nan))
    print 'getting error'
    print 'izipped',izipped
    for int in range(len(izipped)):
        # results.append(np.nanstd(izipped[int]))
        results.append(stats.sem(izipped[int]))
    return results
