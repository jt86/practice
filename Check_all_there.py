# import os
# path = '/Volumes/LocalDataHD/j/jt/jt306/Desktop/results/GPC_conf'
# count=0
# for num_selected in [300,500]:
#     for datasetnum in range(49):
#         for seed in range(10):
#             for fold in range(10):
#                 location = 'error-100-{}-tech-{}-{}-{}-.txt'.format(num_selected, datasetnum, seed, fold)
#                 full_path = os.path.join(path,location)
#                 if not os.path.exists(full_path):
#                     print ('print ("{} {} {} {}")'.format(num_selected, datasetnum, seed, fold))
#                     count+=1
# print (count)


import os
path = '/Volumes/LocalDataHD/j/jt/jt306/Desktop/finalresults'
count=0

for root, dirs, files in os.walk(path, topdown=True):
    for name in files:
        print (name)
        with open(os.path.join(root, name)) as file:
            parameters = [int(item) for item in (file.readlines()[2])[:-1].split()]
            print (parameters)
            num_selected,datasetnum,seed,fold=parameters[0],parameters[1],parameters[2],parameters[3]

        with open(os.path.join(root, name)) as file:
            score= (file.readlines()[-1][4:])
            print (score)

        location = 'error-100-{}-tech-{}-{}-{}-.txt'.format(num_selected, datasetnum, seed, fold)
        with open (os.path.join(root, location)) as newfile:
            newfile.write(score)
# for num_selected in [300,500]w:
#     for datasetnum in range(49):
#         for seed in range(10):
#             for fold in range(10):
#                 location = 'error-100-{}-tech-{}-{}-{}-.txt'.format(num_selected, datasetnum, seed, fold)
#                 full_path = os.path.join(path,location)
#                 if not os.path.exists(full_path):
#                     print ('print ("{} {} {} {}")'.format(num_selected, datasetnum, seed, fold))
#                     count+=1
print (count)