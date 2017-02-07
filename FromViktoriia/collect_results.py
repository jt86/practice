#!/usr/bin/python
import sys
import os
import pdb
import numpy
import scipy.stats

PATH = ""

def collect_ACC(method, all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset):

        method_ACC = numpy.zeros( (len(all_data), len(all_classes)) )
        for d_ind, data_id in enumerate(all_data):
                for c_ind,class_id in enumerate(all_classes):
                        method_output = PATH_methods + dataset + "_%s_%sclass_%ddata_ACC.txt"%(method, class_id, data_id)

                        # AwA_privfeat_rfe_top_0.50top_3class_3data_ACC.txt
                        # AwA_svm_                     0class_1data_ACC.txt'
                        # method_ACC[d_ind,c_ind] = numpy.loadtxt(method_output)

        ACC_mean = numpy.mean(method_ACC, axis=0)
        ACC_mean_std = numpy.r_[[numpy.mean(method_ACC, axis=0)], [numpy.std(method_ACC, axis=0)]]
        ACC_mean_std_err = numpy.r_[[numpy.mean(method_ACC, axis=0)], [numpy.std(method_ACC, axis=0)/numpy.sqrt(len(all_data))]]
        numpy.savetxt(PATH_output + dataset + "_"+ method + "_ACC_mean_std.txt", ACC_mean_std, fmt='%f')
        numpy.savetxt(PATH_output_ACC + dataset + "_"+ method + "_ACC.txt", method_ACC, fmt='%f')
        numpy.savetxt(PATH_output + dataset + "_"+ method + "_ACC_mean_std_err.txt", ACC_mean_std_err, fmt='%f')
        return ACC_mean, ACC_mean_std_err



def print_tofile_rest(file_name,  output, all_classes, test_classes, text=''):
        f = open(file_name,'w')
	print>>f, text
        for c_ind,class_id in enumerate(all_classes):
                pair = ' %s'%class_id
                value = ''
                for j in xrange(output.shape[1]):
                        value=value + '%f\t '%output[c_ind,j]
                print>>f, value  + pair
        f.close()

def main_binary_AwA(dataset, all_classes, all_data, test_classes, PATH_methods, PATH_output, PATH_output_ACC):

        output1, std1=collect_ACC("svm", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output2, std2=collect_ACC("privfeat_rfe_top_0.05top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output2_, std2_=collect_ACC("privfeat_rfe_top_SVMRFE_0.05top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output3, std3=collect_ACC("privfeat_rfe_top_0.10top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output3_, std3_=collect_ACC("privfeat_rfe_top_SVMRFE_0.10top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output4, std4=collect_ACC("privfeat_rfe_top_0.25top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output4_, std4_=collect_ACC("privfeat_rfe_top_SVMRFE_0.25top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output5, std5=collect_ACC("privfeat_rfe_top_0.50top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output5_, std5_=collect_ACC("privfeat_rfe_top_SVMRFE_0.50top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output6, std6=collect_ACC("privfeat_rfe_top_0.75top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output6_, std6_=collect_ACC("privfeat_rfe_top_SVMRFE_0.75top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

	output=numpy.r_[[output1],[output2],[output3],[output4],[output5],[output6]]
	methodnames='SVM\t\t privfeatRFE 0.05\t privfeatRFE 0.10\t privfeatRFE 0.25\t privfeatRFE 0.50\t privfeatRFE 0.75\t'

        print ("Priv features RFE 0.05 - SVM", numpy.sum([output2-output1]))
        print ("Priv features RFE 0.10 - SVM", numpy.sum([output3-output1]))
        print ("Priv features RFE 0.25 - SVM", numpy.sum([output4-output1]))
        print ("Priv features RFE 0.50 - SVM", numpy.sum([output5-output1]))
        print ("Priv features RFE 0.75 - SVM", numpy.sum([output6-output1]))

        return output,methodnames

def main_binary_arcene(dataset, all_classes, all_data, test_classes, PATH_methods, PATH_output, PATH_output_ACC):

        output1, std1=collect_ACC("svm", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output2, std2=collect_ACC("privfeat_rfe_top_0.05top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output2_, std2_=collect_ACC("privfeat_rfe_top_SVMRFE_0.05top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output3, std3=collect_ACC("privfeat_rfe_top_0.10top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output3_, std3_=collect_ACC("privfeat_rfe_top_SVMRFE_0.10top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output4, std4=collect_ACC("privfeat_rfe_top_0.20top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output4_, std4_=collect_ACC("privfeat_rfe_top_SVMRFE_0.20top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output5, std5=collect_ACC("privfeat_rfe_top_0.30top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output5_, std5_=collect_ACC("privfeat_rfe_top_SVMRFE_0.30top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output6, std6=collect_ACC("privfeat_rfe_top_0.40top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output6_, std6_=collect_ACC("privfeat_rfe_top_SVMRFE_0.40top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output7, std7=collect_ACC("privfeat_rfe_top_0.50top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output7_, std7_=collect_ACC("privfeat_rfe_top_SVMRFE_0.50top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)

        output8, std8=collect_ACC("privfeat_rfe_top_0.75top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)
        output8_, std8_=collect_ACC("privfeat_rfe_top_SVMRFE_0.75top", all_classes, all_data, PATH_methods, PATH_output, PATH_output_ACC, dataset)


	output=numpy.r_[[output1],[output2],[output3],[output4],[output5],[output6],[output7],[output8]]
	methodnames='SVM\t\t privfeatRFE 0.05\t privfeatRFE 0.10\t privfeatRFE 0.25\t privfeatRFE 0.50\t privfeatRFE 0.75\t'

        print ("Priv features RFE 0.05 - SVM", numpy.sum([output2-output1]))
        print ("Priv features RFE 0.10 - SVM", numpy.sum([output3-output1]))
        print ("Priv features RFE 0.2 - SVM", numpy.sum([output4-output1]))
        print ("Priv features RFE 0.3 - SVM", numpy.sum([output5-output1]))
        print ("Priv features RFE 0.4 - SVM", numpy.sum([output6-output1]))
        print ("Priv features RFE 0.5 - SVM", numpy.sum([output7-output1]))
        print ("Priv features RFE 0.75 - SVM", numpy.sum([output8-output1]))

        return output,methodnames


def main(dataset):

        PATH_methods = PATH + dataset + "/Results/"
        PATH_output = PATH + dataset + "/Output_main/"
        PATH_output_ACC = PATH + dataset + "/Output_main/ACC/"

        if not os.path.exists(PATH_output):
                os.makedirs(PATH_output)
        if not os.path.exists(PATH_output_ACC):
                os.makedirs(PATH_output_ACC)

        #==========AwA==========
        if dataset=="AwA":
                all_classes = ['0','1','2','3','4','5','6','7','8','9']
                all_data = [1,2,3,4,5,6,7,8,9,10]
                test_classes = ["Chimpanzee", "Giant panda", "Leopard", "Persian cat", "Pig", "Hippopotamus", "Humpback whale", "Raccoon", "Rat", "Seal"]

                output,methodnames = main_binary_AwA('AwA', all_classes, all_data, test_classes, PATH_methods, PATH_output, PATH_output_ACC)
                print_tofile_rest(PATH_output + "_AwA.txt", output.T, all_classes, test_classes,methodnames)

	#==========Arcene==========
        if dataset=="arcene":
                all_classes = ['01']
                all_data = [1,2,3,4,5,6,7,8,9,10]
                test_classes = ["Positive"]

                output,methodnames = main_binary_arcene('arcene', all_classes, all_data, test_classes, PATH_methods, PATH_output, PATH_output_ACC)
                print_tofile_rest(PATH_output + "_arcene.txt", output.T, all_classes, test_classes,methodnames)


if __name__ == '__main__':

    main("AwA")
	#main("arcene")  
                                                                                                                                              


