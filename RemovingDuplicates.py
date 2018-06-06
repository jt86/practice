import os, sys
from Get_Full_Path import get_full_path
import csv

folder = get_full_path('/Users/joe/Desktop/Privileged_Data/AllResults/tech/linear/lufe-svmplus-mi-300selected-top100priv-unfixed')
new_folder = get_full_path('/Users/joe/Desktop/Privileged_Data/AllResults/tech/linear/lufe-svmplus-mi-300selected-top100priv')
print(folder)

for datasetnum in range(295):
    full_path = folder+'/tech{}/lufe-1.csv'.format(datasetnum)
    os.mkdir(new_folder + '/tech{}'.format(datasetnum))
    new_full_path = new_folder+'/tech{}/lufe-1.csv'.format(datasetnum)
    with open(full_path, 'r') as inp, open(new_full_path, 'w') as out:
        writer = csv.writer(out)
        print(datasetnum)
        for count, row in enumerate(csv.reader(inp)):
            if count<10:
                print(count,row)
                writer.writerow(row)