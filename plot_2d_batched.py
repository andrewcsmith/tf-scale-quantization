from harmonicity import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python import debug as tf_debug

c = 0.05
n_points = 101

print("UNFINISHED")

# with n_points = 101, parallel
# TotalSeconds      : 4.5721547

run_options = tf.RunOptions()
run_options.report_tensor_allocations_upon_oom = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

dimensions = 2
log_pitches = tf.get_variable("log_pitches", [n_points**dimensions, dimensions], dtype=tf.float64)
init_op = tf.global_variables_initializer()

vectors = vector_space_graph(5, 4, name="vectors")
z_op = calc_func_graph(log_pitches, vectors, c=c)

sess.run(init_op)

xs = np.linspace(0.0, 1.0, n_points)
ys = np.linspace(0.0, 1.0, n_points)
xv, yv = np.meshgrid(xs, ys, sparse=False)
starting_coordinates = np.array([xv, yv]).reshape(dimensions, n_points**dimensions).T

zv = np.empty((0, starting_coordinates.shape[1]))

for row in np.split(starting_coordinates, starting_coordinates.shape[0]):
    sess.run([log_pitches.assign(row)])
    new_row = sess.run(z_op, options=run_options).reshape(n_points, n_points)
    zv = np.concatenate((zv, new_row), axis=0)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xv, yv, zv, rstride=1, cstride=1)
plt.show()
