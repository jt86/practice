__author__ = 'jt306'

import os, sys
import argparse
from MainFunction import main_function
from Arcene import get_arcene_data
from Gisette import get_gisette_data
from Get_Full_Path import get_full_path
from Madelon import get_madelon_data
from Dorothea import get_dorothea_data
from Vote import get_vote_data
from Heart import get_heart_data
from Haberman import get_haberman_data
from Crx import get_crx_data
from Mushroom import get_mushroom_data
from Hepatitis import get_hepatitis_data
from Cancer import get_cancer_data
from Bankruptcy import get_bankruptcy_data
from Spambase import get_spambase_data
from Musk2 import get_musk2_data
from Musk1 import get_musk1_data
from Ionosphere import get_ionosphere_data
from HillValley import get_hillvalley_data
from Wine import get_wine_data

import time

if __name__ == '__main__':

    print "runnning"

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('--input', type=str, required=True,
                        choices=('arcene', 'awa', 'madelon', 'gisette', 'dorothea', 'vote', 'heart', 'haberman', 'crx',
                                 'mushroom', 'hepatitis','cancer','bankruptcy', 'spambase', 'musk2', 'musk1', 'heart2',
                                 'ionosphere', 'hillvalley', 'wine'),
                        help='name of input data')
    parser.add_argument('--debug',action='store_true',
                        help='whether to speed things up and cut corners')
    parser.add_argument('--num-folds', type=int, required=True,
                        help='number of folds for cross-validation', default=5)
    # parser.add_argument('--tuple', type=int, required=True,
    #                     help='values of n to consider: min max and increment size', nargs = 3)
    # parser.add_argument('--c-values', type = float, required = True, nargs = '+',
    #                 help = 'list of C values to do parameter estimation over')

    parser.add_argument('--peeking', dest='peeking', action='store_true',
                        help = 'whether or not parameter estimation is performed peeking at the test data each fold')

    parser.add_argument('--rank-metric', type=str, required=True,
                        choices = ('f','c', 'r', 'r2'), help='the method used to rank features')
    parser.add_argument('--prop-priv', type = int, required=True,
                        help = 'the inverse of the proportion of non-selected features to use as privileged')
    parser.add_argument('--gamma-multiplier', type = int,
                        help = 'the factor by which to multiply gamma for the SVM+')
    parser.add_argument('--bottom-n-percent', type = int,
                    help = 'the percentage of worst-ranked features to reject')

    # parser.set_defaults(feature=True)


    args = parser.parse_args()


    arcene = False
    t0 = time.clock()
    if args.input == 'arcene':
        if args.debug == True:
            print 'debug = true'
            features_array, labels_array = get_arcene_data(debug=True)
        else:
            print 'debug = false'
            features_array, labels_array = get_arcene_data()
        tuple = (919,9920,1000)


    elif args.input == 'gisette':
        features_array, labels_array = get_gisette_data()
    elif args.input == 'madelon':
        features_array, labels_array = get_madelon_data()
    elif args.input == 'dorothea':
        features_array, labels_array = get_dorothea_data()
    elif args.input == 'vote':
        features_array, labels_array = get_vote_data()
        tuple = [1,16,1]
    elif args.input == 'heart' or args.input == 'heart2':
        features_array, labels_array = get_heart_data()
        tuple = [1,13,1]
    elif args.input == 'haberman':
        features_array, labels_array = get_haberman_data()
    elif args.input == 'crx':
        features_array, labels_array = get_crx_data()
        tuple = [1,42,5]
    elif args.input == 'mushroom':
        features_array, labels_array = get_mushroom_data()
    elif args.input == 'hepatitis':
        features_array, labels_array = get_hepatitis_data()
    elif args.input == 'cancer':
        features_array, labels_array = get_cancer_data()
        tuple = [15,153,15]
    elif args.input == 'bankruptcy':
        features_array, labels_array = get_bankruptcy_data()
        tuple = [1,18,1]
    elif args.input == 'spambase':
        features_array, labels_array = get_spambase_data()
    elif args.input == 'musk2':
        features_array, labels_array = get_musk2_data()
    elif args.input == 'musk1':
        features_array, labels_array = get_musk1_data()
        tuple = [16,166,16]
    elif args.input == 'ionosphere':
        features_array, labels_array = get_ionosphere_data()
        tuple = [3,34,3]
    elif args.input == 'hillvalley':
        features_array, labels_array = get_hillvalley_data()
        tuple = [9,101,10]
    elif args.input == 'wine':
        features_array, labels_array = get_wine_data()
        tuple = [1,13,1]
    else:
        raise ValueError('WTF is that input')

    print "all args", args

    print "time taken to load data:", time.clock()-t0

    print "tuple",tuple

    keyword = str(args.input)+"_peeking=" +str(args.peeking)+"_"+str(args.num_folds)+"-folds_"+str(args.rank_metric)+"_rejected-"+str(args.bottom_n_percent)+"%-used-gamma_times_"+str(args.gamma_multiplier)
    print"\n\n"+keyword+"\n\n"
    print keyword
    all_results_directory =  get_full_path('privileged_data/results/')
    # output_directory = (os.path.join(all_results_directory,args.output_dir))
    output_directory = (os.path.join(all_results_directory,keyword))


    print "output directory", output_directory



    if not os.path.exists(output_directory):
        os.mkdir(output_directory)

    # print 'peeking=',args.peeking

    c_values = [0.1,1.,10.]

    #todo : change c_values in main function back to args.cvalues!!!

    main_function(features_array, labels_array, output_directory, args.num_folds, tuple, c_values,
                  peeking=args.peeking, dataset=args.input, rank_metric = args.rank_metric, prop_priv = args.prop_priv,
                  multiplier = args.gamma_multiplier, bottom_n_percent=args.bottom_n_percent)


    #
    #     prefix = '/home/j/jt/jt306'

    #     prefix = '/Volumes/LocalDataHD/jt306'




