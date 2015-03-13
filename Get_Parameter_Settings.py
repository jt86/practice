__author__ = 'jt306'

import random
values = [0.01, 0.1, 1., 10., 100.]
values2 = [item/10000 for item in values]
print values2

inputs = ['arcene', 'bankruptcy', 'cancer','crx','dexter','gisette','haberman','heart','hillvalley','ionosphere',
          'madelon','mushroom','musk1','musk2','spambase','vote','wine']

metrics = ['f','c','r','r2']
c_values = [0.1,1.,10.]
c_values = [value for value in c_values]
bottom_n_values = [0, 25, 50]

pattern = '--input {} --num-folds {} --rank-metric {} --c-values {} --prop-priv {} --gamma-multiplier {} --peeking {} --bottom-n-percent {}'


i = 0
with open('all_experiment_parameters_%d.txt'%random.randint(0, 9999), 'w') as outf:
    for input in inputs:
        for metric in metrics:
            for bottom_n in bottom_n_values:
                txt = pattern.format(input, 5, metric, c_values, 1, 1, True ,bottom_n )
                outf.write(txt)
                outf.write('\n')
                i += 1
                print txt





