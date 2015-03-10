import numpy as np
import csv
import sys
dict={}

dict[0] = ['b','c','x','f','k','s']
dict[1] = ['f','g','y','s']
dict[2] = ['n','b','c','g','r','p','u','e','w','y']
dict[3] = ['t','f']
dict[4] = ['a','l','c','y','f','m','n','p','s']
dict[5] = ['a', 'd','f','n']
dict[6] = ['c','w','d']
dict[7] = ['b','n']
dict[8] = ['k','n','b','h','g','r','o','p','u','e','w','y']
dict[9] = ['e','t']
dict[10] = ['b','c','u','e','z','r','?']
dict[11] = ['f','y','k','s']
dict[12] = ['f','y','k','s']
dict[13] = ['n','b','c','g','o','p','e','w','y']
dict[14] = ['n','b','c','g','o','p','e','w','y']
dict[15] = ['p','u']
dict[16] = ['n','o','w','y']
dict[17] =['n','o','t']
dict[18] = ['c','e','f','l','n','p','s','z']
dict[19] = ['k','n','b','h','r','o','u','w','y']
dict[20] = ['a','c','n','s','v','y']
dict[21]= ['g','l','m','p','u','w','d']


def get_mushroom_data():
    with open("/Volumes/LocalDataHD/jt306/Desktop/Privileged_Data/new_data/mushroom.data.csv", "r+") as infile:
        features_array = []
        reader = csv.reader(infile,dialect=csv.excel_tab)
        for row in reader:
            features_array.append(str(row).translate(None,"[]'").split(","))


    features_array = np.array(features_array)
    labels_array = np.array(features_array[:,0],dtype=int)
    labels_array = np.reshape(labels_array,8124)
    #
    # print 'forst label',(labels_array[0])
    # print 'type of first label', (type(labels_array[0]))
    # print 'mushroom labels:',labels_array
    #
    # sys.exit(0)
    # labels_array[labels_array=='e']=1
    # labels_array[labels_array=='p']=-1

    print labels_array

    features_array = features_array[:,1:]

    new_features_array = []
    for instance in features_array:
        new_features_array.append(convert_instance(instance))

    features_array = np.array(new_features_array)
    print features_array.shape
    # print 'labels', labels_array
    return (features_array, labels_array)







def convert_to_binary(instance, feature_number, list):
    binary = np.zeros(len(list))
    value = str(instance[feature_number])
    if value != '?':
        # print 'value',value
        binary[list.index(value)]=1
    # print 'binary',binary
    return binary



def convert_instance(instance):
    output_so_far = []
    for i in range(22):
        # print "output so far", output_so_far
        list = dict[i]
        new_binary = convert_to_binary(instance,i,list)
        output_so_far = np.concatenate((output_so_far,new_binary))
    return output_so_far



# first_instance = get_mushroom_data()[0]
# print "First instance", first_instance
# print convert_instance(first_instance)

# get_mushroom_data()
# features_array, labels_array = get_mushroom_data()
# print features_array[50]