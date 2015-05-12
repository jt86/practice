import argparse
from SingleFold2 import single_fold

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataset', type=str, required=True, help='name of input data')
    # parser.add_argument('--datasetnum', type=int, required=True, help='id number for AwA')
    parser.add_argument('--num-folds', type=int, required=True,
                        help='number of folds for cross-validation')

    parser.add_argument('--peeking', dest='peeking', action='store_true',
                        help='whether or not parameter estimation is performed peeking at the test data each fold')

    parser.add_argument('--cmin', type=int, required = True, help='power of lowest value for c (bottom end of log range)')
    parser.add_argument('--cmax', type=int, required = True, help='power of highest value for c (top of log range)')
    parser.add_argument('--cstarmin',  help = 'power of lowest value for cstar')
    parser.add_argument('--cstarmax',  help = 'power of highest value for cstar')
    parser.add_argument('--numberofcs', type=int, help = 'the number of values to investigate for c and c*')

    parser.add_argument('--kernel', type=str)#, choices = ('rbf','linear'))

    parser.add_argument('--k', type=int, required = True)


    args = parser.parse_args()
    print 'input is', args.dataset
    print ' all args',args

    # dataset='awa{}'.format(args.dataset_num)

    single_fold(k=args.k, num_folds=args.num_folds, dataset=args.dataset, peeking=args.peeking,
                kernel=args.kernel, cmin=args.cmin,cmax=args.cmax,number_of_cs=args.numberofcs,
                cstarmin=args.cstarmin, cstarmax=args.cstarmax)

