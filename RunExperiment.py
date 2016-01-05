import argparse
from SingleFoldSlice import single_fold
# from GetMI import single_fold


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataset', type=str, required=True, help='name of input data')

    parser.add_argument('--topk', type=int, required=True, help='num of top features selected for use as normal info')
    # parser.add_argument('--datasetnum', type=int, required=True, help='id number for AwA')
    # parser.add_argument('--num-folds', type=int, required=True,
    #                     help='number of folds for cross-validation')

    # parser.add_argument('--peeking', dest='peeking', action='store_true',
    #                     help='whether or not parameter estimation is performed peeking at the test data each fold')

    parser.add_argument('--cmin', type=int, required = True, help='power of lowest value for c (bottom end of log range)')
    parser.add_argument('--cmax', type=int, required = True, help='power of highest value for c (top of log range)')
    parser.add_argument('--numberofcs', type=int, help = 'the number of values to investigate for c and c*')

    parser.add_argument('--kernel', type=str)#, choices = ('rbf','linear'))

    parser.add_argument('--k', type=int, required = True)
    parser.add_argument('--datasetnum', type=int, required = True)

    parser.add_argument('--skfseed', type=int, required=True, help='seed for random division of SKF - to allow 10x10fold')

    parser.add_argument('--percentofpriv', type=int, required=True, help='percentage of priv info to take')

    parser.add_argument('--percentageofinstances', type=int, required=True, help='percentage of training instances used')

    args = parser.parse_args()
    print('input is', args.dataset)
    print(' all args',args)

    # dataset='awa{}'.format(args.dataset_num)
    # single_fold(k=args.k, dataset=args.dataset, percentage=args.percentage,
    #             kernel=args.kernel, cmin=args.cmin,cmax=args.cmax,number_of_cs=args.numberofcs)


    single_fold(k=args.k, dataset=args.dataset, topk=args.topk, datasetnum =args.datasetnum,
                kernel=args.kernel, cmin=args.cmin,cmax=args.cmax,number_of_cs=args.numberofcs, skfseed=args.skfseed,
                percent_of_priv = args.percentofpriv)