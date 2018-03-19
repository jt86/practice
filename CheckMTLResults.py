from Get_Full_Path import get_full_path
import os

folder = get_full_path('Desktop/MTLResults/')
list_nums = [1,2,3]+[item for item in range(10,300,10)]+[item for item in range (1000,2200,100)]

num_hidden_units = 3200
rate = 0.0001
weight = 1
count = 1
count_exists, count_not = 0, 0
for featsel in ('anova','bahsic','chi2','mi','rfe'):

    for num_feats in list_nums:
        for num_unsel_feats in list_nums:
            # print(num_unsel_feats)
            for fold in range(10):
                fh = os.path.join(folder,'MTL_{}_results/MTLresultsfile-3200units-weight1.0-numfeats={}-learnrate0.0001-fold{}.csv')\
                    .format(featsel,num_unsel_feats,fold)
                if not os.path.exists(fh):
                    print("print('--numhiddenunits 3200 --rate 0.0001 --weight 1 --featselector {} --foldnum {} --numunselfeats {}')" \
                          .format(featsel, fold, num_unsel_feats))
                    count_not+=1
                if os.path.exists(fh):
                    count_exists+=1
    print(count_exists,count_not)
#
# for featsel in ('anova','bahsic','chi2','mi','rfe'):
#     for foldnum in range(10):
#         print("print('--numhiddenunits 3200 --rate 0.0001 --weight 1 --featselector {} --foldnum {} --numunselfeats all')" \
#             .format(featsel, fold))