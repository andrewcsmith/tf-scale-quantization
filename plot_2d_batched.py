from harmonicity import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python import debug as tf_debug

c = 0.05
# 10-cent increments
n_points = 128

# with n_points = 101, parallel
# TotalSeconds      : 4.5721547

run_options = tf.RunOptions()
run_options.report_tensor_allocations_upon_oom = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)

dimensions = 2
batch_size = 1024
log_pitches = tf.get_variable("log_pitches", [batch_size, dimensions], dtype=tf.float64)
vectors = vector_space_graph(5, 4, bounds=(0.0, 1.0), name="vectors")

init_op = tf.global_variables_initializer()
sess.run(init_op)

xs = np.linspace(0.0, 1.0, n_points)
ys = np.linspace(0.0, 1.0, n_points)
xv, yv = np.meshgrid(xs, ys, sparse=False)
zv = np.array([])

# Creates a vector of pairs of shape [n_points**dimensions, dimensions]
starting_coordinates = np.array([xv, yv]).reshape(dimensions, n_points**dimensions).T
starting_dataset = tf.data.Dataset.from_tensor_slices({
    "coords": tf.constant(starting_coordinates)
})
starting_iterator = starting_dataset.batch(batch_size).make_one_shot_iterator()
next_element = starting_iterator.get_next()

while True:
    try:
        with tf.control_dependencies([log_pitches.assign(next_element['coords'])]):
            z_op = calc_func_graph(log_pitches, vectors, c=c, bounds=tf.constant([[0.0, 0.5], [0.5, 1.0]], dtype=tf.float64))
        new_points = sess.run(z_op)
        zv = np.concatenate([zv, new_points])
    except tf.errors.OutOfRangeError:
        break

# print(zv)
zv = np.expand_dims(zv, 0).reshape([n_points, n_points])
# print(zv.shape)
# print(zv)

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_surface(xv, yv, zv, rstride=1, cstride=1)
plt.show()
