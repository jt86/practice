__author__ = 'jt306'
print('getting parameter settings')

#inputs = ['heart', 'vote', 'wine', 'bankruptcy', 'ionosphere', 'crx','hillvalley', 'cancer',  'musk1','arcene', 'gisette', 'haberman','madelon']

# inputs = ['madelon','dexter','arcene','gisette','dorothea']
# # inputs = ['dexter']
# bottom_n_values = [0]
# peeking_allowed = False
# pattern = '--input {} --num-folds 5 --rank-metric r2 --prop-priv 1 --bottom-n-percent 0 --cmin -4 --cmax 1 --cstarmin 0 --cstarmax 5 --numberofcs 6 --kernel rbf --taket {}'
#
# #--input wine --num-folds 3 --cmin -3 --cmax 3 --numberofcs 2 --rank-metric r2 --prop-priv 1 --gamma-multiplier 1 --bottom-n-percent 0 --peeking
#
# for input in inputs:
#     for peeking in ['--peeking','']:
#         txt = pattern.format(input, peeking)
#         print txt


    #
    # k = eval(sys.argv[1])
    # class_id = sys.argv[2]		#class_id depending on the dataset AwA:'0','1',...,'9'; arcene:'01'
    # N = eval(sys.argv[3])		#N depending on the dataset
    # test_N=eval(sys.argv[4])	#test_N depending on the dataset
    # method = sys.argv[5]
    # topK=eval(sys.argv[6])		#top features(fraction of features) to select for privfeat_rfe; for example, 0.05, 0.1, 0.25, 0.5, 0.75
    # dataset = sys.argv[7]


N=10
test_N=20
method='privfeat_rfe_top'
dataset='AwA'

for k in range(10):
    for class_id in range(10):
        for topK in [0.05, 0.10, 0.25, 0.50, 0.75]:
            txt = k, class_id, N, test_N, method, topK, dataset
            print txt