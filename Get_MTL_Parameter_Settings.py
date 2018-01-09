
num_hidden_units = 3200
rate = 0.0001
weight = 1
featsel='chi2'

count = 1

for foldnum in range(10):
    for num_unsel_feats in [1,2,3]+[item for item in range(10,300,10)]+[item for item in range (1000,2200,100)]+['all']:
        print('--numhiddenunits {} --rate {} --weight {} --featselector {} foldnum {} --numunselfeats {}'\
        .format(num_hidden_units, rate, weight, featsel, foldnum, num_unsel_feats))
        count+=1
print (count)
