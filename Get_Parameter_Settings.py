__author__ = 'jt306'


print('getting parameter settings')

inputs = ['heart']#, 'vote', 'wine', 'bankruptcy', 'ionosphere', 'crx','hillvalley', 'cancer',  'musk1','arcene', 'gisette', 'haberman','madelon']

bottom_n_values = [0]
peeking_allowed = False
pattern = '--input {} --num-folds 5 --rank-metric r2 --prop-priv 1 --gamma-multiplier 1 --bottom-n-percent 0 --cmin -5 --cmax 0 --cstarmin 0 --cstarmax 5 --numberofcs 7 {}'

#--input wine --num-folds 3 --cmin -3 --cmax 3 --numberofcs 2 --rank-metric r2 --prop-priv 1 --gamma-multiplier 1 --bottom-n-percent 0 --peeking
list = [1,2,3]

for input in inputs:
    for peeking in ['--peeking','']:
        txt = pattern.format(input, peeking)
        print txt