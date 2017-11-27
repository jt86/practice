from GetSingleFoldData import get_train_and_test_this_fold
from SingleFoldSlice import Experiment_Setting
import tensorflow as tf
import numpy as np

s = Experiment_Setting(foldnum=9, topk=30, dataset='tech', datasetnum=0, kernel='linear',
         cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=10, take_top_t='top', lupimethod='svmplus',
         featsel='rfe',classifier='lufe',stepsize=0.1)
x_tr, x_te, y_tr, y_te =  (get_train_and_test_this_fold(s))

print([item.shape for item in [x_tr, x_te, y_tr, y_te]])
y_tr = y_tr.reshape((y_tr.shape[0],1))
y_te = y_te.reshape((y_te.shape[0],1))

print(y_tr.shape)


feat_dims = x_tr.shape[1]
output_dims = 1
num_nodes = 2



# Define the Placeholders
X = tf.placeholder("float", [None, feat_dims], name="X")
Y1 = tf.placeholder("float", [None, output_dims], name="Y1")
Y2 = tf.placeholder("float", [None, output_dims], name="Y2")

# Define the weights for the layers
initial_shared_layer_weights = np.random.rand(feat_dims, num_nodes)
initial_Y1_layer_weights = np.random.rand(num_nodes,output_dims)
initial_Y2_layer_weights = np.random.rand(num_nodes,output_dims)


print(initial_shared_layer_weights)

shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")



# Construct the Layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X,shared_layer_weights))
task1_layer_output = tf.nn.relu(tf.matmul(shared_layer, Y1_layer_weights))
task2_layer_output = tf.nn.relu(tf.matmul(shared_layer, Y2_layer_weights))

# Calculate Loss
task1_loss = tf.nn.l2_loss(Y1 - task1_layer_output)
task2_loss = tf.nn.l2_loss(Y2 - task2_layer_output)
Joint_Loss = task1_loss + task2_loss

accuracy1 = tf.metrics.accuracy(Y1,task1_layer_output)
accuracy2 = tf.metrics.accuracy(Y2,task2_layer_output)

# optimisers
Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
Y1_op = tf.train.AdamOptimizer().minimize(task1_loss)
Y2_op = tf.train.AdamOptimizer().minimize(task2_loss)



# Joint Training
# Calculation (Session) Code
# ==========================

# open the session

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    Optimiser, loss_score, accuracy = sess.run([Optimiser, Joint_Loss, accuracy1],{X:x_tr, Y1:y_tr, Y2:y_tr})

    tf.Print(task1_layer_output,[task1_layer_output])

    # Y1.eval()
    # task1_layer_output.eval()
    print('Y1')
    tf.Print(Y1,[Y1])

    print(loss_score)
    print(tf.trainable_variables())
    loss_score2 = sess.run(Joint_Loss, {X: x_te, Y1: y_te, Y2: y_te})#, shared_layer_weights: shared_layer_weights})
    print (loss_score2)

    correct = tf.equal(Y1,task1_layer_output)
    print(correct)

    # accuracy_score = sess.run(accuracy1,{X:x_te,Y1:y_te,Y2:y_te})