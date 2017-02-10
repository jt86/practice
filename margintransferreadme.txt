This is the Python implementation of the Margin Transfer method described in the arXiv paper:
V.Sharmanska, N.Quadrianto, C.H. Lampert: "Learning to Transfer Privileged Information", arXiv:1410.0389v1, 2014.

Margin Transfer is an adaptation of the Rank Transfer method proposed for the ranking framework in: 
V.Sharmanska, N.Quadrianto, C.H. Lampert: "Learning to Rank using Privileged Information", ICCV 2013.

If you use this code or Rank Transfer code, please, cite accordingly. 

=====Instructions=====
1. Download and compile the LIBLINEAR library with instance weight support

download from here: http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/#weights_for_data_instances; 
go to the directory with python bindings ('python/') and compile (for Unix systems, type 'make' in the terminal window; otherwise read the manual). 

2. Place the script margintransfer.py (together with the 'data/' folder and the script margintransfer_cv.py) 
into the folder with python bindings ('python/'). 
One could also export PYTHONPATH to be able to run the codes from anywhere. 

3. To run the demo, simply type 'python margintransfer.py' in the terminal window. 
In our demo, we load a running example with L1-normalized SURF features as image data (X) and DAP predicted attributes as privileged data (X*) for solving the object classification task in images (Y is binary). 

Additionally, we also provide the 5x5 cross validation model selection procedure for Margin Transfer (in margintransfer_cv.py). 
It can be run during the training procedure by uncommenting two lines of code: line 18 and line 40 (in margintransfer.py).

