__author__ = 'jt306'


values = [0.01, 0.1, 1., 10., 100.]
values2 = [item/10000 for item in values]
print values2

pattern = '--c {} --cstar {} --gamma{} --gammastar{} --output-dir {} --input {}'
i = 0
with open('all_experiment_parameters.txt', 'w') as outf:
    for c in values:
        for cstar in values:
            for gamma in values2:
                for gammastar in values2:
                    txt = pattern.format(c, cstar, gamma, gammastar, 'output%d' % i)  #hardcoded to output to directory with number of iteration
                    outf.write(txt)
                    outf.write('\n')
                    i += 1



