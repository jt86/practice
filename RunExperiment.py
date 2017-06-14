import argparse
# from SingleFoldSliceUnivariate import single_fold
from SingleFoldSlice import single_fold, Experiment_Setting
# from GetMI import single_fold# from SingleFold_dSVM_normalised import single_fold
# from SingleFoldUnivariate import single_fold
# from SingleFoldSlice import save_instance_and_feature_indices_for_R, save_dataset_for_R
from pprint import pprint

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--dataset', type=str, help='name of input data')

    parser.add_argument('--topk', type=int, help='num of top features selected for use as normal info')
    # parser.add_argument('--datasetnum', type=int, required=True, help='id number for AwA')
    # parser.add_argument('--num-folds', type=int, required=True,
    #                     help='number of folds for cross-validation')

    # parser.add_argument('--peeking', dest='peeking', action='store_true',
    #                     help='whether or not parameter estimation is performed peeking at the test data each fold')

    parser.add_argument('--cmin', type=int,  help='power of lowest value for c (bottom end of log range)')
    parser.add_argument('--cmax', type=int,  help='power of highest value for c (top of log range)')
    parser.add_argument('--numberofcs', type=int, help = 'the number of values to investigate for c and c*')

    # parser.add_argument('--cvalues', type=str)

    parser.add_argument('--kernel', type=str)#, choices = ('rbf','linear'))

    parser.add_argument('--foldnum', type=int)
    parser.add_argument('--datasetnum', type=int)

    parser.add_argument('--skfseed', type=int,  help='seed for random division of SKF - to allow 10x10fold')

    parser.add_argument('--percentofpriv', type=int,  help='percentage of priv info to take')

    parser.add_argument('--percentageofinstances', type=int,  help='percentage of training instances used')

    parser.add_argument('--taketopt',type=str,help='if top: take top percent ofpriv.if bottom, take bottom')

    parser.add_argument('--featsel', required=False,
                        help='metric for univariate')

    parser.add_argument('--lupimethod', type=str, required=False, help='which lupi method to use')

    parser.add_argument('--classifier', type=str, required=False, help = 'whether its featselection, baseline, svm+ etc')

    # parser.add_argument('--dSVMC',type=str,required=True,help='fixed C parameter for the SVM used to get d-values')

    args = parser.parse_args()
    print('input is', args.dataset)
    print(' all args',args)

    # dataset='awa{}'.format(args.dataset_num)
    # single_fold(k=args.k, dataset=args.dataset, percentage=args.percentage,
    #             kernel=args.kernel, cmin=args.cmin,cmax=args.cmax,number_of_cs=args.numberofcs)


    s = Experiment_Setting(foldnum=args.foldnum, topk=args.topk, dataset=args.dataset, datasetnum=args.datasetnum,
                           kernel=args.kernel, cmin = args.cmin, cmax= args.cmax, numberofcs=args.numberofcs, skfseed=args.skfseed,
                           percent_of_priv=args.percentofpriv, percentageofinstances=args.percentageofinstances,
                           take_top_t=args.taketopt, lupimethod=args.lupimethod, featsel=args.featsel, classifier=args.classifier)
    single_fold(s)



    # save_instance_and_feature_indices_for_R(k=args.k, dataset=args.dataset, topk=args.topk, datasetnum =args.datasetnum,
    #             kernel=args.kernel, cmin=args.cmin,cmax=args.cmax,number_of_cs=args.numberofcs, skfseed=args.skfseed)


    # save_instance_and_feature_indices_for_R(k=args.k, dataset=args.dataset, topk=args.topk, datasetnum =args.datasetnum,
    #             kernel=args.kernel, cmin=args.cmin,cmax=args.cmax,number_of_cs=args.numberofcs, skfseed=args.skfseed)
    #
    # save_dataset_for_R(args.datasetnum)

    # single_fold(k, topk, dataset, datasetnum, kernel, cmin, cmax, number_of_cs, skfseed, percent_of_priv,
    #             percentageofinstances, take_top_t):