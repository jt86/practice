__author__ = 'jt306'
import numpy as np
import itertools

list_of_lists = [[5,5,5],[43,3],[3,1,5,6,1]]
# lists = np.array(list_of_lists)
# print lists
# print np.mean(lists, axis=1)
# np.nanmean()

# def get_mean_of_lists(list_of_lists):
#     max_length = len(max(enumerate(list_of_lists), key = lambda l: len(l[1]))[1])
#     list_of_means = np.zeros(max_length)
#     for l in list_of_lists:
#         print 'length', len(l)
#         for index, item in enumerate (l):
#             list_of_means[index]+=[item]
#     return list_of_means/len(list_of_lists)
#
# print get_mean_of_lists(list_of_lists)


#
# def get_mean_of_lists(list_of_lists):
#     max_length = len(max(enumerate(list_of_lists), key = lambda l: len(l[1]))[1])
#     # list_of_means = np.empty (3)
#     # print list_of_means
#     list_of_means = np.empty(5)
#     # list_of_means = [[]]*max_length
#     for l in list_of_lists:
#         print 'length', len(l)
#         for index, item in enumerate (l):
#             list_of_means[index]+=[item]
#             print list_of_means
#     return list_of_means
#
# print get_mean_of_lists(list_of_lists)

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
        results.append(np.nanstd(izipped[int]))
    return results
