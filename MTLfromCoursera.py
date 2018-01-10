import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
# from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from GetSingleFoldData import get_train_and_test_this_fold
from SingleFoldSlice import Experiment_Setting, get_norm_priv
import sys
np.random.seed(1)
import os
from Get_Full_Path import get_full_path

# def convert_to_one_hot(Y, C):
#     Y = np.eye(C)[Y.reshape(-1)].T
#     return Y

def convert_to_one_hot(y_tr,C):
    y_new = np.zeros((len(y_tr),2))
    for i in range(len(y_tr)):
        if y_tr[i] == 1:
            y_new[i]=[1,0]
        else:
            y_new[i]=[0,1]
    y_new = y_new.T
    return y_new


def get_tech_data(s,num_unsel_feats=300):

    print(s.datasetnum)
    x_train, x_test, y_train, y_test =  get_train_and_test_this_fold(s)
    normal_train, normal_test, priv_train, priv_test = get_norm_priv(s,x_train,x_test)
    print(priv_train.shape[1])
    if num_unsel_feats == 'all':
        num_unsel_feats = priv_train.shape[1]
    print(num_unsel_feats)
    # x_tr, x_te = normal_train[:,:400], normal_test[:,:400]
    y_tr1 = convert_to_one_hot(y_train,2)
    y_te1 = convert_to_one_hot(y_test,2)
    y_tr2=priv_train[:,:num_unsel_feats]
    y_te2=priv_test[:,:num_unsel_feats]
    y_tr2=y_tr2.reshape(num_unsel_feats,y_tr2.shape[0])
    y_te2=y_te2.reshape(num_unsel_feats,y_te2.shape[0])
    x_tr = normal_train.T
    x_te = normal_test.T
    return(x_tr,x_te,y_tr1,y_te1,y_tr2,y_te2)


def random_mini_batches(X, Y1, Y2, mini_batch_size=64, seed=0):

    m = X.shape[1]  # number of training examples used to crete permuation, which is used as index to shuffle

    mini_batches = []
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y1 = Y1[:, permutation].reshape((Y1.shape[0], m))
    shuffled_Y2 = Y2[:, permutation].reshape((Y2.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    print('mini_batch_size', mini_batch_size, 'num_complete_minibatches',num_complete_minibatches)
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y1 = shuffled_Y1[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y2 = shuffled_Y2[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y1, mini_batch_Y2)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y1 = shuffled_Y1[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y2 = shuffled_Y2[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y1, mini_batch_Y2)
        mini_batches.append(mini_batch)

    return mini_batches


def create_placeholders(n_x, n_y1, n_y2):
    X = tf.placeholder("float", shape=[n_x, None], name='X_placeholder')
    Y1 = tf.placeholder("float", shape=[n_y1, None], name='Y_placeholder')
    Y2 = tf.placeholder("float", shape=[n_y2, None], name='Y_placeholder')
    return X, Y1, Y2


def initialize_parameters(dims, num_unsel_feats, num_hidden_units):
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [num_hidden_units, dims], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [num_hidden_units, 1], initializer=tf.zeros_initializer())
    W2task1 = tf.get_variable("W2task1", [2, num_hidden_units], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2task1 = tf.get_variable("b2task1", [2, 1], initializer=tf.zeros_initializer())
    W2task2 = tf.get_variable("W2task2", [num_unsel_feats, num_hidden_units], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2task2 = tf.get_variable("b2task2", [num_unsel_feats, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1, "b1": b1, "W2task1": W2task1, "b2task1": b2task1, "W2task2":W2task2, "b2task2":b2task2}
    return parameters


def forward_propagation(X, parameters):
    W1, b1 = parameters['W1'],parameters['b1']
    W2task1, b2task1 = parameters['W2task1'], parameters['b2task1']
    W2task2, b2task2 = parameters['W2task2'], parameters['b2task2']
    Z1 = tf.add(tf.matmul(W1, X), b1)
    A1 = tf.nn.relu(Z1)
    Z3task1 = tf.add(tf.matmul(W2task1, A1), b2task1)
    Z3task2 = tf.add(tf.matmul(W2task2, A1), b2task2)
    return Z3task1, Z3task2


def compute_cost(Z3a, Z3b, Y1, Y2,num_unsel_feats, task_2_weight=1):
    logitsa = tf.transpose(Z3a)
    labels1 = tf.transpose(Y1)
    cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logitsa, labels=labels1))
    cost2 = tf.nn.l2_loss(Y2 - Z3b)
    # tf.shape(logitsa)
    cost = cost1+(task_2_weight*cost2/num_unsel_feats)
    return cost

tf.reset_default_graph()


def model(setting, X_train, Y_train1, Y_train2, X_test, Y_test1, Y_test2, dims, num_unsel_feats, results_file, learning_rate=0.001,
          num_epochs=50, minibatch_size=32, print_cost=True, task_2_weight=1,num_hidden_units=320):

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y1 = Y_train1.shape[0]  # n_y : output size
    n_y2 = Y_train2.shape[0]  # n_y : output size
    print(n_y2)
    costs = []  # To keep track of the cost
    X, Y1, Y2 = create_placeholders(n_x, n_y1, n_y2)
    parameters = initialize_parameters(dims, n_y2, num_hidden_units)
    Z3a, Z3b = forward_propagation(X, parameters) #fwd prop
    cost = compute_cost(Z3a, Z3b, Y1, Y2,n_y2,task_2_weight)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #bck prop
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            print(X_train, Y_train1, Y_train2, minibatch_size, seed)
            minibatches = random_mini_batches(X_train, Y_train1, Y_train2, minibatch_size, seed)
            for minibatch in minibatches:
                (minibatch_X, minibatch_Y1, minibatch_Y2) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y1: minibatch_Y1, Y2: minibatch_Y2})
                epoch_cost += minibatch_cost / num_minibatches
            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)
        # lets save the parameters in a variable
        parameters = sess.run(parameters)

        # Calculate the correct predictions
        correct_prediction1 = tf.equal(tf.argmax(Z3a), tf.argmax(Y1))
        # correct_prediction2 = tf.equal(tf.argmax(Z3b), tf.argmax(Y2))
        # print("cost eval",compute_cost.eval({Z3a:Z3a, Z3b:Z3b, Y1:Y1, Y2:Y2}))
        # print(correct_prediction1,correct_prediction2)
        # Calculate accuracy on the test set
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
        # accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

        print("Train Accuracy task 1:", accuracy1.eval({X: X_train, Y1: Y_train1}))
        print("Test Accuracy task 1:", accuracy1.eval({X: X_test, Y1: Y_test1}))
        # print("Train Accuracy task 2:", accuracy2.eval({X: X_train, Y2: Y_train2}))
        # print("Test Accuracy task 2:", accuracy2.eval({X: X_test, Y2: Y_test2}))


        results_file.write("Dataset,{},Fold,{},Weight,{},Train,{}, Test,{}\n".format(setting.datasetnum,setting.foldnum,task_2_weight,
                                 accuracy1.eval({X: X_train, Y1: Y_train1}),accuracy1.eval({X: X_test, Y1: Y_test1})))

        return parameters

# num_hidden_units = 3200
# rate = 0.0001
# weight = 1
# featsel='chi2'

# for num_unsel_feats in [item for item in range (1000,2200,100)]:
# # for num_unsel_feats in [1,2,3]+[item for item in range(10,300,10)]:
# # for num_unsel_feats in ['all']:
#     print ('num unsel feats',num_unsel_feats)
#     with open(get_full_path('Desktop/Privileged_Data/MTL_{}_results/MTLresultsfile-{}units-weight{}-numfeats={}-learnrate{}.csv'
#                                     .format(featsel.upper(),num_hidden_units,weight,num_unsel_feats,rate)), 'a') as results_file:
#         for datasetnum in range(295):
#             print('\n')
#             s = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
#                                    cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
#                                    take_top_t='top', lupimethod='svmplus',
#                                    featsel=featsel, classifier='lufe', stepsize=0.1)
#             x_tr, x_te, y_tr1, y_te1, y_tr2, y_te2=get_tech_data(s,num_unsel_feats)
#             dims = x_tr.shape[0]
#             parameters = model(s, x_tr, y_tr1, y_tr2, x_te, y_te1, y_te2,  dims, num_unsel_feats,results_file,task_2_weight=weight,num_hidden_units=num_hidden_units, learning_rate=rate)

def run_mtl(num_hidden_units, rate, weight, featsel, num_unsel_feats, foldnum, num_datasets=295):
    for datasetnum in range(num_datasets):
        with open(get_full_path('Desktop/Privileged_Data/MTL_{}_results/MTLresultsfile-{}units-weight{}-numfeats={}-learnrate{}-fold{}.csv'
                                        .format(featsel.upper(),num_hidden_units,weight,num_unsel_feats,rate,foldnum)), 'a') as results_file:
            s = Experiment_Setting(foldnum=foldnum, topk=300, dataset='tech', datasetnum=datasetnum, kernel='linear',
                                   cmin=-3, cmax=3, numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100,
                                   take_top_t='top', lupimethod='svmplus',
                                   featsel=featsel, classifier='lufe', stepsize=0.1)
            x_tr, x_te, y_tr1, y_te1, y_tr2, y_te2 = get_tech_data(s, num_unsel_feats)
            dims = x_tr.shape[0]
            parameters = model(s, x_tr, y_tr1, y_tr2, x_te, y_te1, y_te2, dims, num_unsel_feats, results_file,
                               task_2_weight=weight, num_hidden_units=num_hidden_units, learning_rate=rate)


# run_mtl(3200, 0.0001, 1, 'chi2', 10, 1)