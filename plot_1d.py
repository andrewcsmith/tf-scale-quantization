from harmonicity import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

c = 0.05
n_points = 1001

# import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_pitches = tf.get_variable("log_pitches", [n_points, 1], dtype=tf.float64)
init_op = tf.global_variables_initializer()

vectors = vector_space_graph(5, 4, name="vectors")
y_op = calc_func_graph(log_pitches, vectors, c=c)

sess.run(init_op)

xs = np.linspace(0.0, 1.0, n_points)
ys = sess.run([log_pitches.assign(tf.expand_dims(xs, 1)), y_op])[1]
# ys = np.array([sess.run([log_pitches.assign([x]), y_op])[1] for x in xs])

plt.plot(xs, ys)
plt.show()
