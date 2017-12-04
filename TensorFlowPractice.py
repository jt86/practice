from GetSingleFoldData import get_train_and_test_this_fold
from SingleFoldSlice import Experiment_Setting
import tensorflow as tf
import numpy as np
import sys





s = Experiment_Setting(foldnum=9, topk=30, dataset='tech', datasetnum=0, kernel='linear',
         cmin=-3,cmax=3,numberofcs=7, skfseed=1, percent_of_priv=100, percentageofinstances=10, take_top_t='top', lupimethod='svmplus',
         featsel='rfe',classifier='lufe',stepsize=0.1)
x_tr, x_te, y_tr, y_te =  (get_train_and_test_this_fold(s))

#
# np.save('x_tr',x_tr[:,:100])
# np.save('x_te',x_te[:,:100])
# np.save('y_tr',y_tr)
# np.save('y_te',y_te)

x_tr = np.load('x_tr.npy')
x_te = np.load('x_te.npy')
y_tr = np.load('y_tr.npy')
y_te = np.load('y_te.npy')
# print([item.shape for item in [x_tr, x_te, y_tr, y_te]])

y_tr = y_tr.reshape((y_tr.shape[0],1))
y_te = y_te.reshape((y_te.shape[0],1))
#
# y_tr_new = np.zeros((len(y_tr),2))
# for i in range(len(y_tr)):
#     if y_tr[i] == 1:
#         y_tr_new[i]=[1,0]
#     else:
#         y_tr_new[i]=[0,1]
#
#
# y_te_new = np.zeros((len(y_te),2))
# for i in range(len(y_te)):
#     if y_tr[i] == 1:
#         y_te_new[i]=[1,0]
#     else:
#         y_te_new[i]=[0,1]
#
# print(y_tr.shape, y_tr_new.shape)
# print(y_te.shape, y_te_new.shape)
#
# np.save('y_te',y_te_new)
# np.save('y_tr',y_tr_new)

for item in [y_te,y_tr]:
    item[item==-1]=0

print(y_tr.shape,y_te.shape)

# sys.exit()
feat_dims = x_tr.shape[1]
output_dims = 1
num_shared_nodes = 20

# Define the Placeholders
X = tf.placeholder("float", [None, feat_dims], name="X")
Y1 = tf.placeholder("float", [None, 1], name="Y1")
Y2 = tf.placeholder("float", [None, 1], name="Y2")




# Define the weights for the layers
initial_shared_layer_weights = np.random.rand(feat_dims, num_shared_nodes)
initial_Y1_layer_weights = np.random.rand(num_shared_nodes, output_dims)
initial_Y2_layer_weights = np.random.rand(num_shared_nodes, output_dims)


shared_layer_weights = tf.Variable(initial_shared_layer_weights, name="share_W", dtype="float32")
Y1_layer_weights = tf.Variable(initial_Y1_layer_weights, name="share_Y1", dtype="float32")
Y2_layer_weights = tf.Variable(initial_Y2_layer_weights, name="share_Y2", dtype="float32")



# Construct the Layers with RELU Activations
shared_layer = tf.nn.relu(tf.matmul(X,shared_layer_weights))
# task1_layer_output = tf.nn.sigmoid(tf.matmul(shared_layer, Y1_layer_weights))
# task2_layer_output = tf.nn.sigmoid(tf.matmul(shared_layer, Y2_layer_weights))
task1_layer_output = (tf.matmul(shared_layer, Y1_layer_weights))
task2_layer_output = (tf.matmul(shared_layer, Y2_layer_weights))


# Make predicitons by taking the max value from the softmax output
task_1_predictions = tf.argmax(task1_layer_output, 1)
task_2_predictions = tf.argmax(task2_layer_output, 1)


# Calculate Loss
task1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y1, logits=task1_layer_output)
task2_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels = Y2, logits=task2_layer_output)
# task2_loss = tf.nn.l2_loss(Y2-task2_layer_output)

Joint_Loss = task1_loss + task2_loss

# optimisers
Optimiser = tf.train.AdamOptimizer().minimize(Joint_Loss)
# Y1_op = tf.train.AdamOptimizer().minimize(task1_loss)
# Y2_op = tf.train.AdamOptimizer().minimize(task2_loss)

# get accuracy
accuracy1 = tf.metrics.accuracy(Y1,task_1_predictions)

# Joint Training
# Calculation (Session) Code
# ==========================

# open the session

with tf.Session() as sess:
    # sess.run(tf.global_variables_initializer())
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)



    test_accuracy = sess.run([accuracy1],{X:x_te, Y1:y_te, Y2:y_te})
    print('test acc',test_accuracy)


    Optimiser, loss_score, train_accuracy, task_1_values, task_1_preds,task_2_preds \
        = sess.run([Optimiser, Joint_Loss, accuracy1, task1_layer_output, task_1_predictions, task_1_predictions],{X:x_tr, Y1:y_tr, Y2:y_tr})


    print(tf.Print(task1_layer_output,[task1_layer_output]))


    # print('task 1 values',task_1_values)
    # print('task1 predictions',task_1_preds)
    # print('task2 predictions',task_2_preds)
    # print(task_1_preds-task_2_preds)


    print('loss score',loss_score)
    print('train_accuracy', train_accuracy)
    print('trainable vars', tf.trainable_variables())

    test_accuracy = sess.run([accuracy1],{X:x_te, Y1:y_te, Y2:y_te})
    print('test acc',test_accuracy)
    #
    # loss_score2,  = sess.run(Joint_Loss, {X: x_te, Y1: y_te, Y2: y_te})#, shared_layer_weights: shared_layer_weights})



    correct = tf.equal(Y1,task1_layer_output)
    print(correct)

    # accuracy_score = sess.run(accuracy1,{X:x_te,Y1:y_te,Y2:y_te})