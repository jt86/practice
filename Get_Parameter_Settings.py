__author__ = 'jt306'
import logging

logging.info('getting parameter settings')

inputs = ['arcene', 'bankruptcy', 'cancer', 'crx', 'gisette', 'haberman', 'heart', 'hillvalley', 'ionosphere',
          'madelon', 'musk1', 'musk2', 'spambase', 'vote', 'wine']

metrics = ['f', 'c', 'r2']

inputs = ['heart']

bottom_n_values = [0, 25, 50]
peeking_allowed = True
pattern = '--input {} --num-folds {} --rank-metric {} --prop-priv {} --gamma-multiplier {} --bottom-n-percent {}'

for input in inputs:
    for metric in metrics:
        for bottom_n in bottom_n_values:
            txt = pattern.format(input, 5, metric, 1, 1, bottom_n)
            if peeking_allowed:
                txt += ' --peeking'
            logging.info(txt)





