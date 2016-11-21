import os
path = '/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/techtc300_preprocessed'
count=0

#49 datasets were originally used. This goes through all the datasets and saves to file those 246 that were NOT


list_of_49=[]
for item in open(os.path.join(path,'techtc_files_49.txt')).readlines():
    print (item[:-1])
    list_of_49.append(item[:-1])


list_of_new_data=[]
count=0

for root, dirs, files in os.walk('/Volumes/LocalDataHD/j/jt/jt306/Desktop/Privileged_Data/techtc300_preprocessed', topdown=True):
    name = (root.split('/')[-1])
    if name not in list_of_49 and 'Exp' in name:
        count+=1
        print (count,name)
        list_of_new_data.append(name)
        with open(os.path.join(path,'techtc_files_246.txt'),'a') as filename:
            filename.write(name+'\n')


print (len(list_of_new_data))