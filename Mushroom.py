import numpy as np
import csv
import sys
import logging

dict = {}

dict[0] = ['b', 'c', 'x', 'f', 'k', 's']
dict[1] = ['f', 'g', 'y', 's']
dict[2] = ['n', 'b', 'c', 'g', 'r', 'p', 'u', 'e', 'w', 'y']
dict[3] = ['t', 'f']
dict[4] = ['a', 'l', 'c', 'y', 'f', 'm', 'n', 'p', 's']
dict[5] = ['a', 'd', 'f', 'n']
dict[6] = ['c', 'w', 'd']
dict[7] = ['b', 'n']
dict[8] = ['k', 'n', 'b', 'h', 'g', 'r', 'o', 'p', 'u', 'e', 'w', 'y']
dict[9] = ['e', 't']
dict[10] = ['b', 'c', 'u', 'e', 'z', 'r', '?']
dict[11] = ['f', 'y', 'k', 's']
dict[12] = ['f', 'y', 'k', 's']
dict[13] = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']
dict[14] = ['n', 'b', 'c', 'g', 'o', 'p', 'e', 'w', 'y']
dict[15] = ['p', 'u']
dict[16] = ['n', 'o', 'w', 'y']
dict[17] = ['n', 'o', 't']
dict[18] = ['c', 'e', 'f', 'l', 'n', 'p', 's', 'z']
dict[19] = ['k', 'n', 'b', 'h', 'r', 'o', 'u', 'w', 'y']
dict[20] = ['a', 'c', 'n', 's', 'v', 'y']
dict[21] = ['g', 'l', 'm', 'p', 'u', 'w', 'd']



numerical_value_dict = {'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 'f':6, 'g':7, 'h':8, 'i':9, 'j':10, 'k':11, 'l':12, 'm':13, 'n':14, 'o':15,
        'p':16, 'q':17, 'r':18, 's':19, 't':20, 'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26}




def get_mushroom_data():
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/mushroom.data.csv", "r+") as infile:
        features_array = []
        reader = csv.reader(infile, dialect=csv.excel_tab)
        for row in reader:
            features_array.append(str(row).translate(None, "[]'").split(","))

    features_array = np.array(features_array)
    labels_array = np.array(features_array[:, 0], dtype=int)
    labels_array = np.reshape(labels_array, 8124)
    #
    # logging.info( 'forst label',(labels_array[0]))
    # logging.info( 'type of first label', (type(labels_array[0])))
    # logging.info( 'mushroom labels:',labels_array)
    #
    # sys.exit(0)
    # labels_array[labels_array=='e']=1
    # labels_array[labels_array=='p']=-1

    print(features_array.shape, labels_array.shape)

    logging.info(labels_array)

    features_array = features_array[:, 1:]

    new_features_array = []
    # for instance in features_array:
    #     new_features_array.append(convert_instance(instance))

    features_array = np.array(new_features_array)
    logging.info(features_array.shape)
    # logging.info( 'labels', labels_array)
    return (features_array, labels_array)

get_mushroom_data()

def convert_to_binary(instance, feature_number, list):
    binary = np.zeros(len(list))
    value = str(instance[feature_number])
    if value != '?':
        # logging.info( 'value',value)
        binary[list.index(value)] = 1
    # logging.info( 'binary',binary)
    return binary


def convert_instance(instance):
    output_so_far = []
    for i in range(22):
        # logging.info( "output so far", output_so_far)
        list = dict[i]
        new_binary = convert_to_binary(instance, i, list)
        output_so_far = np.concatenate((output_so_far, new_binary))
    return output_so_far



    # first_instance = get_mushroom_data()[0]
    # logging.info( "First instance", first_instance)
    # logging.info( convert_instance(first_instance))

    # get_mushroom_data()
    # features_array, labels_array = get_mushroom_data()
    # logging.info( features_array[50])