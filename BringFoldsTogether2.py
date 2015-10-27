__author__ = 'jt306'
import os,sys
import numpy as np
from Get_Figures2 import get_figures, get_mean_and_error
import os
from Get_Full_Path import get_full_path
from matplotlib import pyplot as plt


def bring_folds_together(num_folds,cross_validation_folder):#,cross_validation_folder2):

    all_folds_SVM, all_folds_LUPI = np.zeros((num_folds,len(list_of_values))),np.zeros((num_folds,len(list_of_values)))
    for i in range(num_folds):
        for percent_num, percentage in enumerate(list_of_values):

            cv_svm_file = open(os.path.join(cross_validation_folder,'svm-{}-{}.csv').format(i,percentage),'r')
            svm_score = float(cv_svm_file.readline().split(',')[0])
            all_folds_SVM[i,percent_num]=(svm_score)

            cv_lupi_file = open(os.path.join(cross_validation_folder,'lupi-{}-{}.csv').format(i,percentage),'r')
            lupi_score = float(cv_lupi_file.readline().split(',')[0])
            all_folds_LUPI[i,percent_num]=(lupi_score)



    # all_folds_LUPI_top =  np.zeros((num_folds,len(list_of_values)))
    # for i in range(num_folds):
    #     for percent_num, percentage in enumerate(list_of_values):
    #
    #         cv_lupi_top_file = open(os.path.join(cross_validation_folder2,'lupi-{}-{}.csv').format(i+1,percentage),'r')
    #         lupi_score_top = float(cv_lupi_top_file.readline().split(',')[0])
    #         all_folds_LUPI_top[i,percent_num]=(lupi_score_top)


    cv_baseline_file = open(os.path.join(cross_validation_folder,'baseline.csv'),'r')
    baseline_list = cv_baseline_file.readline().split(',')[:-1]
    baseline_list = list(map(float, baseline_list))



    return all_folds_SVM, all_folds_LUPI, baseline_list#, all_folds_LUPI_top



def read_from_disk_and_plot(num_folds, cross_validation_folder, list_of_values, x_axis_list,dataset,graph_directory, datasetnum):#, cross_validation_folder2):
    all_folds_SVM, all_folds_LUPI, baseline_score = [],[],[]
    for i in range(10):
        all_folds_SVM[i], all_folds_LUPI[i], baseline_score[i]  = bring_folds_together(num_folds,cross_validation_folder)#cross_validation_folder2)w
    #all_folds_LUPI_top
    baseline_results = [baseline_score] * len(list_of_values)
    # print ('baseline score',np.mean(baseline_score))

    print ('all folds svm', np.mean(all_folds_SVM,axis=0)[0])

    get_figures(x_axis_list, all_folds_SVM, all_folds_LUPI, baseline_results, dataset, graph_directory, datasetnum)
    #all_folds_LUPI_top
    return(np.mean(baseline_score),np.mean(all_folds_SVM,axis=0)[0])


######################### CHANGE THIS PART TO GET RESULTS

# list_of_values = list(range(100,1001,100))
list_of_values = [300,500]


x = list(range(49))
y = list(range(49))
plt.plot(x,y)
plt.savefig('testplot')
# sys.exit()



list_of_baselines=[]
list_of_300_rfe=[]
dataset='tech'

baseline_scores_list,top300scores_list = [],[]
for i in range(49):
    print ('doing dataset',i)
    for j in range(10):
        output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/10x10tech/ALL/fixedC-10fold-{}-{}-RFE-baseline-step=0.1-percent_of_priv=100').format(dataset,i))
        # output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/STEP10percent66-33-LONGWORDS-FIXEDC/ALL/not-fixedC-33-66-tech-{}-RFE-baseline-step=0.1-percent_of_priv=100').format(i))
        # output_directory2 = os.path.join(get_full_path('Desktop/Privileged_Data/66-33/TOP50/not-fixedC-33-66-tech-{}-RFE-baseline-step=1000-percent_of_priv=50').format(i))
        graph_directory = (get_full_path('Desktop/Privileged_Data/10fold/ALLplots'))
        cross_validation_folder = os.path.join(output_directory,'cross-validation')
        # cross_validation_folder2 = os.path.join(output_directory2,'cross-validation')
        baseline_score, top300score = read_from_disk_and_plot(10,cross_validation_folder,list_of_values,list_of_values,dataset,graph_directory,i)#, cross_validation_folder2)
        list_of_baselines.append(1-baseline_score)
        list_of_300_rfe.append(1-top300score)
print (list_of_baselines)

# THIS PART TO GET 49-dataset plot
# print ('length',list_of_300_rfe)
# print ('length',list_of_300_rfe[0])
#
# fig = plt.figure()
# ax2 = fig.add_subplot(111)
# fig.suptitle('TechTC-300 - Error rates', fontsize=20)
# line1 = ax2.plot(list(range(49)),list_of_baselines,'b',label='All features')
# line2 = ax2.plot(list(range(49)),list_of_300_rfe,'r',label='RFE - top 300 features')
# ax2.legend()#([line1,line2],['All features',['RFE - top 300 features']])
# fig.savefig('newplot')




    # all_folds_SVM, all_folds_LUPI, baseline_score = bring_folds_together(num_folds,cross_validation_folder, graph_directory)
    # results, errors = get_mean_and_error(all_folds_SVM)w
    # zipped = list(zip(results,errors))
    # topks=list_of_values
    # for item in zip(topks,zipped):
    #     print('&{1:.3f}$\pm${2:.3f}'.format(item[0],item[1][0],item[1][1]))
#
# list_of_values = [5,10,25,50,75]
# for dataset in ['madelon','arcene','dorothea','dexter']:
#     output_directory = os.path.join(get_full_path('Desktop/Privileged_Data/NIPS/not-fixedC-25-75-{}-0-RFE-baseline-step=1000').format(dataset))
#     graph_directory = get_full_path('Desktop/Privileged_Data/NIPS/')
#     cross_validation_folder = os.path.join(output_directory,'cross-validation')
#     read_from_disk_and_plot(10,cross_validation_folder,list_of_values,list_of_values,dataset,graph_directory,0)