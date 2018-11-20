from harmonicity import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

c = 0.05
n_points = 512

# In sequence at n_points = 101
# TotalSeconds      : 16.0295673
# In parallel
# TotalSeconds      : 4.6289348

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_pitches = tf.get_variable("log_pitches", [n_points, 1], dtype=tf.float64)
init_op = tf.global_variables_initializer()

vectors = vector_space_graph(5, 4, name="vectors")
y_op = calc_func_graph(log_pitches, vectors, c=c)

sess.run(init_op)

xs = np.linspace(0.0, 3.0, n_points)
_, ys = sess.run([log_pitches.assign(xs[:, None]), y_op])
# The following iterates over each element in sequence, rather than loading it
# all as a single parallel graph.
# ys = np.array([sess.run([log_pitches.assign([x]), y_op])[1] for x in xs])

plt.plot(xs, ys)
plt.show()
