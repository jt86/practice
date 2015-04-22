__author__ = 'jt306'


print('getting parameter settings')

inputs = ['heart', 'vote', 'wine', 'bankruptcy', 'ionosphere', 'crx','hillvalley', 'cancer',  'musk1','arcene', 'gisette', 'haberman',
           'madelon']

# metrics = ['f', 'c', 'r2']
metrics = ['r2']

bottom_n_values = [0]
peeking_allowed = False
pattern = '--input {} --num-folds 5 --rank-metric r2 --prop-priv 1 --gamma-multiplier 1 --bottom-n-percent 0 --cmin -3 --cmax 3 --numberofcs 7 {} --cstarmin {} --cstarmax {}'

#--input wine --num-folds 3 --cmin -3 --cmax 3 --numberofcs 2 --rank-metric r2 --prop-priv 1 --gamma-multiplier 1 --bottom-n-percent 0 --peeking
list = [1,2,3]

for input in inputs:
    for peeking in ['--peeking','']:
        for cstarmin, cstarmax in [[-3,3],[0,6]]:
            txt = pattern.format(input, peeking, cstarmin, cstarmax)
            # if peeking_allowed:
            #     txt += ' --peeking'
            print txt