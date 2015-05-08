Main script: 
run_privfeat.py

getdata.py 
has 2 functions to load data/generate train/test datasets for AwA, arcene

getmethod.py
has a variety of functions for cross validation model selection of baselines

Additionally, 
4 scripts for running SVM+:
SVMplus.py, elefant_exceptions.py, vector.py, generic.py


collect_results.py for collecting the accuracy performance of all methods from the folders AwA/Results, arcene/Results

checkreg.py to check the cross validation model selection results


The structure of folders with datasets, cross validation results and accuracy results needs to be as in here:

-folder named as dataset (for example "AwA/")
-inside there is a folder "CV" for cross validation model selection results (for example "AwA/CV") and a folder "Results" for accuracy results (for example "AwA/Results") 




