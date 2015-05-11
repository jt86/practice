import numpy
import matplotlib.pyplot as plt
import pylab
from matplotlib.backends.backend_pdf import PdfPages
import pdb
import math
    
PATH_outcome = "Results/" 

output1, std1=numpy.loadtxt("Output_main/AwA_svm_ACC_mean_std_err.txt")*100.
output2, std2=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_0.05top_ACC_mean_std_err.txt")*100.
output2_, std2_=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_SVMRFE_0.05top_ACC_mean_std_err.txt")*100.
output3, std3=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_0.10top_ACC_mean_std_err.txt")*100.
output3_, std3_=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_SVMRFE_0.10top_ACC_mean_std_err.txt")*100.
output4, std4=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_0.25top_ACC_mean_std_err.txt")*100.
output4_, std4_=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_SVMRFE_0.25top_ACC_mean_std_err.txt")*100.
output5, std5=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_0.50top_ACC_mean_std_err.txt")*100.
output5_, std5_=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_SVMRFE_0.50top_ACC_mean_std_err.txt")*100.
output6, std6=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_0.75top_ACC_mean_std_err.txt")*100.
output6_, std6_=numpy.loadtxt("Output_main/AwA_privfeat_rfe_top_SVMRFE_0.75top_ACC_mean_std_err.txt")*100.

def plot_rfe(classidx,classname):
	plt.clf()
	all_top=[0.05, 0.10, 0.25, 0.5, 0.75]
	T=len(all_top)

	svmjoint=output1[classidx]*numpy.ones(T)
	svmjoint_std=std1[classidx]*numpy.ones(T)
	
	rfe=numpy.r_[output2[classidx],output3[classidx], output4[classidx], output5[classidx], output6[classidx]]
	rfe_std=numpy.r_[std2[classidx],std3[classidx], std4[classidx], std5[classidx], std6[classidx]]
	
	svm_rfe=numpy.r_[output2_[classidx],output3_[classidx], output4_[classidx], output5_[classidx], output6_[classidx]]
	svm_rfe_std=numpy.r_[std2_[classidx],std3_[classidx], std4_[classidx], std5_[classidx], std6_[classidx]]
	
	names = ['SVM', 'RFE and SVM+', 'SVM+RFE']

	params = {'legend.fontsize': 6, 'legend.linewidth': 0.3}
	plt.title(classname)

	#with plt.xkcd():
	x = [1,2,3,4,5]
		
	p1 = plt.errorbar(x, svmjoint, yerr=svmjoint_std, color = 'g', lw=3)
	p2 = plt.errorbar(x, rfe, yerr=rfe_std, color = 'darkorange', lw=3)
	p3 = plt.errorbar(x, svm_rfe, yerr=svm_rfe_std, color = 'b', lw=3)

	plt.legend([p1,p2,p3], names)
	
	plt.xlim([0.5,5.5])
	middle=int(min(svmjoint))
	plt.ylim([middle-3, middle+3])
	plt.xticks(numpy.arange(1, 6), ['0.05%', '0.10%', '0.25%', '0.50%', '0.75%'], fontsize=10)
	plt.ylabel('Accuracy')

	return 1


if __name__ == '__main__':

	
	test_classes = ["Chimpanzee", "Giant panda", "Leopard", "Persian cat", "Pig", "Hippopotamus", "Humpback whale", "Raccoon", "Rat", "Seal"]
	
	fig_file = PdfPages("PrivRFE.pdf")
	for classidx,classname in enumerate(test_classes):
		plot_rfe(classidx,classname)
		fig_file.savefig()
	fig_file.close()


    

