from harmonicity import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.python import debug as tf_debug
import time

c = 0.04
# 10-cent increments
n_points = 128
LEARNING_RATE = 1.0e-3
CONVERGENCE_THRESHOLD = 2.0 ** -32
MAX_ITERS = 1000000

# Benchmark: 
# Converged at iteration:  652
# time: 5.501294700000001

run_options = tf.RunOptions()
run_options.report_tensor_allocations_upon_oom = True
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

dimensions = 2
batch_size = 128
log_pitches = tf.get_variable("log_pitches", [batch_size, dimensions], dtype=tf.float64)

init_op = tf.global_variables_initializer()
sess.run(init_op)

vectors = vector_space_graph(5, 4, bounds=(0.0, 1.0), name="vectors")

xs = np.linspace(0.00, 1.0, n_points)
ys = np.linspace(0.00, 1.0, n_points)
xv, yv = np.meshgrid(xs, ys, sparse=False)
zv = np.array([])

# Creates a vector of pairs of shape [n_points**dimensions, dimensions]
starting_coordinates = np.array([xv, yv]).reshape(dimensions, n_points**dimensions).T
starting_dataset = tf.data.Dataset.from_tensor_slices({
    "coords": tf.constant(starting_coordinates)
})
starting_iterator = starting_dataset.batch(batch_size).make_one_shot_iterator()
next_element = starting_iterator.get_next()
assign_op = log_pitches.assign(next_element['coords'])

opt = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
loss = cost_func(log_pitches, vectors, c=c, name="loss")
opt_op = opt.minimize(loss, var_list=[log_pitches])
compute_grad_op = opt.compute_gradients(loss, var_list=[log_pitches])
grad_norms_op = [tf.nn.l2_loss(g) for g, v in compute_grad_op]
grad_norm_op = tf.add_n(grad_norms_op, name="grad_norm")
norm_convergence_op = tf.less(grad_norm_op, tf.constant(CONVERGENCE_THRESHOLD, dtype=tf.float64))

with tf.control_dependencies([opt_op]):
    stopping_condition_op = tf.reduce_all(norm_convergence_op)

all_possible_pitches_log = np.empty([0, 2])

while True:
    try:
        # Pull the new iterator element and assign
        sess.run([assign_op])
        start_time = time.clock()

        # Begin the loop (same as optimize_1d)
        for idx in range(MAX_ITERS):
            if (sess.run(stopping_condition_op)):
                print("Converged at iteration: ", idx)
                print("time: %s" % (time.clock() - start_time))
                out_pitches = np.array(sess.run(log_pitches))
                all_possible_pitches_log = np.concatenate([all_possible_pitches_log, out_pitches])
                # print("log pitches: ", out_pitches)
                break
    except tf.errors.OutOfRangeError:
        break

print(all_possible_pitches_log.shape)
log_vectors = vector_pitch_distances(vectors)

print(all_possible_pitches_log)
diffs_to_poles = tf.abs(tf.tile(log_vectors[:, None, None], [1, 1, 2]) - all_possible_pitches_log)
mins = tf.argmin(diffs_to_poles, axis=0)
winner = tf.map_fn(lambda m: tf.map_fn(lambda v: vectors[v], m, dtype=tf.float64), mins, dtype=tf.float64)
winners = sess.run(winner)
print(winners)

def vector_to_ratio(vector):
    primes = PRIMES[:vector.shape[0]]
    num = np.where(vector > 0, vector, np.zeros_like(primes))
    den = np.where(vector < 0, vector, np.zeros_like(primes))
    return (
        np.product(np.power(primes, num)), 
        np.product(primes ** np.abs(den))
    )

all_possible_pitches = set()

for row in winners:
    all_possible_pitches.add(tuple([vector_to_ratio(r) for r in row]))

print(len(all_possible_pitches))
print(sorted(all_possible_pitches, key=lambda r: (r[0][0] / r[0][1], r[1][0] / r[1][1])))