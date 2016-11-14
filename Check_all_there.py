import os
path = '/Volumes/LocalDataHD/j/jt/jt306/Desktop/results/GPC_conf'

for num_selected in [300,500]:
    for datasetnum in range(49):
        for seed in range(10):
            for fold in range(10):
                location = 'error-100-{}-tech-{}-{}-{}-.txt'.format(num_selected, datasetnum, seed, fold)
                full_path = os.path.join(path,location)
                if not os.path.exists(full_path):
                    print ('print ("{} {} {} {}")'.format(num_selected, datasetnum, seed, fold))
