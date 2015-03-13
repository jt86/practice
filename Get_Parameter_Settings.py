__author__ = 'jt306'

values = [0.01, 0.1, 1., 10., 100.]
values2 = [item / 10000 for item in values]

inputs = ['arcene', 'bankruptcy', 'cancer', 'crx', 'dexter', 'gisette', 'haberman', 'heart', 'hillvalley', 'ionosphere',
          'madelon', 'mushroom', 'musk1', 'musk2', 'spambase', 'vote', 'wine']
metrics = ['f', 'c', 'r', 'r2']
bottom_n_values = [0, 25, 50]

pattern = '--input {} --num-folds {} --rank-metric {} --prop-priv {} --gamma-multiplier {} --peeking {} --bottom-n-percent {}'

for input in inputs:
    for metric in metrics:
        for bottom_n in bottom_n_values:
            txt = pattern.format(input, 5, metric, 1, 1, True, bottom_n)
            print txt





