__author__ = 'jt306'
import logging

print('getting parameter settings')

inputs = ['heart', 'vote', 'wine', 'bankruptcy', 'ionosphere', 'crx','hillvalley', 'cancer',  'musk1','arcene', 'gisette', 'haberman',
           'madelon', 'musk2', 'spambase', ]

# metrics = ['f', 'c', 'r2']
metrics = ['r2']

bottom_n_values = [0]
peeking_allowed = False
pattern = '--input {} --num-folds {} --rank-metric {} --prop-priv {} --gamma-multiplier {} --bottom-n-percent {} --cmin {} --cmax {}'


# list = [[-3,3],[-2,2],[0,2],[1,3],[0,4]]
list = [[-1,1],[-3,3]]

for input in inputs:
    for a,b in list:
        txt = pattern.format(input, 5, 'r2', 1, 1, 0,a,b)
        if peeking_allowed:
            txt += ' --peeking'
        print txt