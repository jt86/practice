__author__ = 'jt306'

from FromViktoriia import getdata


import numpy as np

def get_awa_data(PATH, class_id):

    animal_indices = {''}
    dataset = 'AwA'
    PATH_data = PATH + "AwA/data_AwA/"
    train_data, test_data, train_labels, test_labels = getdata.getdata_AwA_one_vs_rest(PATH_data, class_id, 10, 20)
    return train_data, test_data, train_labels, test_labels


# train_data, test_data, train_labels, test_Y = get_awa_data('FromViktoriia/','1')
# print train_data.shape
# print test_data.shape
# print train_labels.shape
# print test_Y.shape
