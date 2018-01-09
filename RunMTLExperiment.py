import argparse
from pprint import pprint



num_hidden_units = 3200
rate = 0.0001
weight = 1
featsel='chi2'

# for num_unsel_feats in [item for item in range (1000,2200,100)]:


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--numhiddenunits', type=int, help='number of units in hidden layer')

    parser.add_argument('--rate', type=float, help='learning rate of neural net')

    parser.add_argument('--weight', type=float,  help='weight of secondary task in error function)')

    parser.add_argument('--featselector', type=str,  help='feature selection method used')

    parser.add_argument('--numunselfeats', type=int, help = 'the number of unselected features')

    parser.add_argument('--foldnum', type=int)

    # parser.add_argument('--numdatasets', type=int, help = 'number of datasets')

    args = parser.parse_args()
    print('input is', args.dataset)
    print(' all args',args)


    s = Experiment_Setting(foldnum=args.foldnum, topk=args.topk, dataset=args.dataset, datasetnum=args.datasetnum,
                           kernel=args.kernel, cmin = args.cmin, cmax= args.cmax, numberofcs=args.numberofcs, skfseed=args.skfseed,
                           percent_of_priv=args.percentofpriv, percentageofinstances=args.percentageofinstances,
                           take_top_t=args.taketopt, lupimethod=args.lupimethod, featsel=args.featsel, classifier=args.classifier, stepsize=args.stepsize)
    single_fold(s)

