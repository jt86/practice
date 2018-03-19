import numpy
import matplotlib.pyplot as plt
import pylab
from matplotlib.backends.backend_pdf import PdfPages
import pdb
import math
    
PATH_outcome = "Results/" 

output1, std1=numpy.loadtxt("Output_main/arcene_svm_ACC_mean_std_err.txt")*100.
output2, std2=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.05top_ACC_mean_std_err.txt")*100.
output2_, std2_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.05top_ACC_mean_std_err.txt")*100.
output3, std3=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.10top_ACC_mean_std_err.txt")*100.
output3_, std3_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.10top_ACC_mean_std_err.txt")*100.
output4, std4=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.20top_ACC_mean_std_err.txt")*100.
output4_, std4_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.20top_ACC_mean_std_err.txt")*100.
output5, std5=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.30top_ACC_mean_std_err.txt")*100.
output5_, std5_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.30top_ACC_mean_std_err.txt")*100.
output6, std6=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.40top_ACC_mean_std_err.txt")*100.
output6_, std6_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.40top_ACC_mean_std_err.txt")*100.
output7, std7=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.50top_ACC_mean_std_err.txt")*100.
output7_, std7_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.50top_ACC_mean_std_err.txt")*100.
output8, std8=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_0.75top_ACC_mean_std_err.txt")*100.
output8_, std8_=numpy.loadtxt("Output_main/arcene_privfeat_rfe_top_SVMRFE_0.75top_ACC_mean_std_err.txt")*100.

def plot_rfe(classname):
	plt.clf()
	all_top=[0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.75]
	T=len(all_top)
	
	svmjoint=output1*numpy.ones(T)
	svmjoint_std=std1*numpy.ones(T)
	
	rfe=numpy.r_[output2,output3, output4, output5, output6, output7,output8]
	rfe_std=numpy.r_[std2,std3, std4, std5, std6, std7, std8]
	
	svm_rfe=numpy.r_[output2_,output3_, output4_, output5_, output6_, output7_,output8_]
	svm_rfe_std=numpy.r_[std2_,std3_, std4_, std5_, std6_, std7_, std8_]
	
	names = ['SVM', 'RFE and SVM+', 'SVM+RFE']

	params = {'legend.fontsize': 6, 'legend.linewidth': 0.3}
	plt.title(classname)

	#with plt.xkcd():
	x = [1,2,3,4,5,6,7]
		
	p1 = plt.errorbar(x, svmjoint, yerr=svmjoint_std, color = 'g', lw=3)
	p2 = plt.errorbar(x, rfe, yerr=rfe_std, color = 'darkorange', lw=3)
	p3 = plt.errorbar(x, svm_rfe, yerr=svm_rfe_std, color = 'b', lw=3)

	#plt.show()
	plt.legend([p1,p2,p3], names)
	
	plt.xlim([0.5,7.5])
	middle=int(min(svmjoint))
	plt.ylim([middle-5, middle+5])
	plt.xticks(numpy.arange(1, 8), ['0.05%', '0.10%', '0.20%', '0.30%', '0.40%','0.50%','0.75%'], fontsize=10)
	plt.ylabel('Accuracy')

	return 1

if __name__ == '__main__':
	
	fig_file = PdfPages("PrivRFE.pdf")
	plot_rfe("Arcene")
	fig_file.savefig()
	fig_file.close()

