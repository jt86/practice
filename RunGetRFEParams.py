num_folds=10

top_k_list = [5,10,25,50,75]
for fold_num in range(num_folds):
    for dataset_num in range(10):
        for top_k in top_k_list:
            dataset='awa{}'.format(dataset_num)
            print '--k {}  --dataset {} --top-k-percent {}'.format(fold_num, dataset, top_k)



#
# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description='Process some integers.')
#
#     parser.add_argument('--dataset', type=str, required=True, help='name of input data')
#     parser.add_argument('--k', type=int, required = True, help='fold number - used to seed')
#     parser.add_argument('--top-k-percent', type=int, required=True, help='percentage of features to take')
#
#     args = parser.parse_args()
#     print 'input is', args.dataset
#     print ' all args',args
#
#     get_rfe_params(args.dataset, args.top_k_percent, args.k)