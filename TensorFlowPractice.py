#  GRAPH CODE
# ============

# Import Tensorflow and Numpy
import tensorflow as tf
import numpy as np
from GetSingleFoldData import get_train_and_test_this_fold
from SingleFoldSlice import Experiment_Setting
s = Experiment_Setting(foldnum=9, topk=30, dataset='tech', datasetnum=0, kernel='linear',
         cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=10, take_top_t='top', lupimethod='svmplus',
         featsel='rfe',classifier='lufe',stepsize=0.1)
x_tr, x_te, y_tr, y_te =  (get_train_and_test_this_fold(s))

print([item.shape for item in [x_tr, x_te, y_tr, y_te]])
y_tr = y_tr.reshape((y_tr.shape[0],1))
y_te = y_te.reshape((y_te.shape[0],1))

print(y_tr.shape)

#  GRAPH CODE
# ============

# Import Tensorflow and Numpy
import tensorflow as tf
import numpy as np

# ======================
# Define the Graph
# ======================
num_instances = x_tr.shape[0]
feat_dims = x_tr.shape[1]
output_dims = 1
num_nodes = 2

# Define the Placeholders
X = tf.placeholder("float", [num_instances, feat_dims], name="X")
Y1 = tf.placeholder("float", [num_instances, output_dims], name="Y1")
Y2 = tf.placeholder("float", [num_instances, output_dims], name="Y2")


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
Y1_layer = tf.nn.relu(tf.matmul(shared_layer,Y1_layer_weights))
Y2_layer = tf.nn.relu(tf.matmul(shared_layer,Y2_layer_weights))

# Calculate Loss
Y1_Loss = tf.nn.l2_loss(Y1-Y1_layer)
Y2_Loss = tf.nn.l2_loss(Y2-Y2_layer)
Joint_Loss = Y1_Loss + Y2_Loss

# optimisers
Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
Y1_op = tf.train.AdamOptimizer().minimize(Y1_Loss)
Y2_op = tf.train.AdamOptimizer().minimize(Y2_Loss)

# Joint Training
# Calculation (Session) Code
# ==========================

# open the session

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    Optimiser, Joint_Loss = session.run([Optimiser, Joint_Loss],
                    {
                        X: x_tr,
                        Y1: y_tr,
                        Y2: y_tr
                      })
    # session.print(shared_layer_weights)
    print(Joint_Loss)