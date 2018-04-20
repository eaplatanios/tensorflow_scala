import tensorflow as tf
import numpy as np
from random import random, seed

seed(22)
weight = random()


def batch(batch_size):
        inp = []
        out = []
        for i in range(batch_size):
                r = random()
                inp += [r]
                out += [weight * r]

        n_inp = np.array(inp)
        n_out = np.array(out)
        return np.reshape(n_inp, [-1, 1]), np.reshape(n_out, [-1, 1])


x = tf.placeholder(tf.float32, shape=[None, 1], name="p2s_inputs")
y_ = tf.placeholder(tf.float32, shape=[None, 1], name="p2s_output")
w = tf.get_variable("p2s_weights", dtype=tf.float32, shape=[1, 1], initializer=tf.zeros_initializer, use_resource=True)
prediction = tf.matmul(x, w)
loss = tf.reduce_mean(tf.square(prediction - y_), name="p2s_loss")
trainOp = tf.train.AdagradOptimizer(1.0).minimize(loss, name="p2s_train_op")

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# - - - - SAVE MODEL - - - - -
saver = tf.train.Saver()
saver.save(sess, "./model-store-python/virgin-my-model-resource-backed")


for i in range(200):
        trainBatch = batch(10000)
        feed = {x: trainBatch[0], y_: trainBatch[1]}
        trainWeight, trainLoss, _ = sess.run([w, loss, trainOp], feed_dict=feed)
        print("[iter {}]weight: {}   -  loss: {}".format(i, trainWeight, trainLoss))


# - - - - SAVE MODEL - - - - -
path_to_checkpoint = saver.save(sess, "./model-store-python/my-model-resource-backed")