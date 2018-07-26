import tensorflow as tf
import numpy as np
import sys
from random import seed
import itertools

seed(22)

flagTrain = ["train", "true", "True", "1"]


def generate_one_data(limit=100):
    for x in itertools.count(1):
        for inp in [x, x * -1]:
            if x == limit:
                return
            out = (inp * 8.254) + 2.555

            n_inp = np.array(inp)
            n_out = np.array(out)
            yield (np.reshape(n_inp, [-1, 1]), np.reshape(n_out, [-1, 1]))


# --- CHECK FLAG ---
if len(sys.argv) != 5:
    print("Error:")
    print("python {} <flag_do_train> <device> <abs_path_virgin_model> <abs_path_trained_model>".format(sys.argv[0]))
    exit(1)

do_train = True if sys.argv[1] in flagTrain else False
device = "CPU:0" if sys.argv[2] == "cpu" else "GPU:0"
abs_path_virgin_model = sys.argv[3]
abs_path_trained_model = sys.argv[4]
print("doTrain: ", do_train)

with tf.device(device):
    x = tf.placeholder(tf.float64, shape=[None, 1], name="p2s_input")
    y_ = tf.placeholder(tf.float64, shape=[None, 1], name="p2s_output")
    w = tf.get_variable("p2s_weights_w", dtype=tf.float64, shape=[1, 1], initializer=tf.zeros_initializer,
                        use_resource=True)
    b = tf.get_variable("p2s_weights_b", dtype=tf.float64, shape=[1, 1], initializer=tf.zeros_initializer,
                        use_resource=True)

    prediction = tf.add(tf.matmul(x, w), b, name="p2s_prediction")

    loss = tf.reduce_mean(tf.square(prediction - y_), name="p2s_loss")
    grads_and_vars = tf.train.AdamOptimizer(0.5).compute_gradients(loss)
    trainOp = tf.train.AdagradOptimizer(0.5).apply_gradients(grads_and_vars, name="p2s_train_op")

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()
    saver = tf.train.Saver()

    # - - - - SAVE MODEL - - -
    saver.save(sess, abs_path_virgin_model)

# - - - - TRAIN - - - - -
if do_train:
    i = 0
    for data in generate_one_data():
        inp = data[0]
        expected = data[1]

        # --- TRAINING ---
        _, predicted = sess.run([trainOp, prediction], feed_dict={x: inp, y_: expected})

        # --- RUN ---
        print("[iter {}] expected/predicted: {}, {}\n".format(i, predicted[0], expected))

        # - - - - SAVE MODEL - - - - -
        saver.save(sess, abs_path_trained_model)
        i += 1
