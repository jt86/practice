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




# sys.exit()

s = Experiment_Setting(foldnum=9, topk=300, dataset='tech', datasetnum=1, kernel='linear',
         cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=100, take_top_t='top', lupimethod='svmplus',
         featsel='rfe',classifier='lufe',stepsize=0.1)
x_tr, x_te, y_tr, y_te =  (get_train_and_test_this_fold(s))

normal_train, normal_test, priv_train, priv_test = get_norm_priv(s,x_tr,x_te)

x_tr, x_te = normal_train, normal_test
#
# x_tr = np.load('x_tr.npy')
# x_te = np.load('x_te.npy')
# y_tr = np.load('y_tr.npy')
# y_te = np.load('y_te.npy')
#
# y_tr = y_tr.reshape((1,y_tr.shape[0]))
# y_te = y_te.reshape((1,y_te.shape[0]))

print(y_tr.shape)

y_tr = convert_to_one_hot(y_tr,2)
y_te = convert_to_one_hot(y_te,2)

print(y_tr.shape)

x_tr = x_tr.T
x_te = x_te.T

dims = x_tr.shape[0]
print(x_tr.shape)
print(x_te.shape)


def random_mini_batches(X, Y1, Y2, mini_batch_size=64, seed=0):

    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y1 = Y1[:, permutation].reshape((Y1.shape[0], m))
    shuffled_Y2 = Y1[:, permutation].reshape((Y2.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
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


def initialize_parameters():
    tf.set_random_seed(1)
    W1 = tf.get_variable("W1", [25, dims], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b1 = tf.get_variable("b1", [25, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [12, 25], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b2 = tf.get_variable("b2", [12, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [2, 12], initializer=tf.contrib.layers.xavier_initializer(seed=1))
    b3 = tf.get_variable("b3", [2, 1], initializer=tf.zeros_initializer())
    ### END CODE HERE ###

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}

    return parameters

tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    print(W1.shape,X.shape)
    ### START CODE HERE ### (approx. 5 lines)                     # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)  # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)  # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)  # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)  # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)  # Z3 = np.dot(W3,Z2) + b3
    ### END CODE HERE ###

    return Z3

tf.reset_default_graph()

with tf.Session() as sess:
    X, Y1, Y2 = create_placeholders(dims, 2, 2)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))


# GRADED FUNCTION: compute_cost

def compute_cost(Z3, Y1, Y2):
    logits = tf.transpose(Z3)
    labels1 = tf.transpose(Y1)
    labels2 = tf.transpose(Y2)
    cost1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels1))
    cost2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels2))
    cost = cost1+cost2
    return cost

tf.reset_default_graph()

# with tf.Session() as sess:
#     X, Y1, Y2 = create_placeholders(dims, 2, 2)
#     parameters = initialize_parameters()
#     Z3 = forward_propagation(X, parameters)
#     cost = compute_cost(Z3, Y1, Y2)
#     print("cost = " + str(cost))


def model(X_train, Y_train1, Y_train2, X_test, Y_test1, Y_test2,  learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):

    ops.reset_default_graph()  # to be able to rerun the model without overwriting tf variables
    tf.set_random_seed(1)  # to keep consistent results
    seed = 3  # to keep consistent results
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y1 = Y_train1.shape[0]  # n_y : output size
    n_y2 = Y_train2.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost
    X, Y1, Y2 = create_placeholders(n_x, n_y1, n_y2)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters) #fwd prop
    cost = compute_cost(Z3, Y1, Y2)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost) #bck prop


    init = tf.global_variables_initializer()
    with tf.Session() as sess:

        # Run the initialization
        sess.run(init)

        # Do the training loop
        for epoch in range(num_epochs):

            epoch_cost = 0.  # Defines a cost related to an epoch
            num_minibatches = int(m / minibatch_size)  # number of minibatches of size minibatch_size in the train set
            seed = seed + 1
            minibatches = random_mini_batches(X_train, Y_train1, Y_train2, minibatch_size, seed)

            for minibatch in minibatches:
                # Select a minibatch
                (minibatch_X, minibatch_Y1, minibatch_Y2) = minibatch
                _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y1: minibatch_Y1, Y2: minibatch_Y2})
                epoch_cost += minibatch_cost / num_minibatches

            if print_cost == True and epoch % 100 == 0:
                print("Cost after epoch %i: %f" % (epoch, epoch_cost))
            if print_cost == True and epoch % 5 == 0:
                costs.append(epoch_cost)

        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction1 = tf.equal(tf.argmax(Z3), tf.argmax(Y1))
        correct_prediction2 = tf.equal(tf.argmax(Z3), tf.argmax(Y2))
        print(correct_prediction1,correct_prediction2)

        # Calculate accuracy on the test set
        accuracy1 = tf.reduce_mean(tf.cast(correct_prediction1, "float"))
        accuracy2 = tf.reduce_mean(tf.cast(correct_prediction2, "float"))

        print("Train Accuracy task 1:", accuracy1.eval({X: X_train, Y1: Y_train1}))
        print("Test Accuracy task 1:", accuracy1.eval({X: X_test, Y1: Y_test1}))
        print("Train Accuracy task 2:", accuracy2.eval({X: X_train, Y2: Y_train2}))
        print("Test Accuracy task 2:", accuracy2.eval({X: X_test, Y2: Y_test2}))

        return parameters



parameters = model(x_tr, y_tr, y_tr, x_te, y_te, y_te)

