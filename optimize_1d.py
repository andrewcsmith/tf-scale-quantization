from harmonicity import *
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python import debug as tf_debug

c = 0.03
n_points = 128
CONVERGENCE_THRESHOLD = 1.0e-9

# 14.0481932 Seconds

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

log_pitches = tf.get_variable("log_pitches", [n_points, 1], dtype=tf.float64)
init_op = tf.global_variables_initializer()

vectors = vector_space_graph(5, 6, bounds=(0.0, 1.0), name="vectors")
y_op = calc_func_graph(log_pitches, vectors, c=c)

sess.run(init_op)

xs = np.linspace(0.0, 1.0, n_points)
inputs = tf.expand_dims(xs, 1)
sess.run(init_op)
sess.run(log_pitches.assign(inputs))

opt = tf.train.GradientDescentOptimizer(learning_rate=1.0e-3)
loss = cost_func(log_pitches, vectors, c=c, name="loss")
opt_op = opt.minimize(loss, var_list=[log_pitches])
compute_grad_op = opt.compute_gradients(loss, var_list=[log_pitches])
grad_norms_op = [tf.nn.l2_loss(g) for g, v in compute_grad_op]
grad_norm_op = tf.add_n(grad_norms_op, name="grad_norm")

with tf.control_dependencies([opt_op]):
    stopping_condition_op = tf.reduce_all(tf.less(grad_norm_op, tf.constant(CONVERGENCE_THRESHOLD, dtype=tf.float64)))

for idx in range(100000):
    if (sess.run(stopping_condition_op)):
        print("Converged at iteration: ", idx)
        out_pitches = sess.run(log_pitches)
        print(out_pitches * 1200.0)
        break

log_vectors = vector_pitch_distances(vectors)
diffs_to_poles = tf.abs(log_vectors - log_pitches)
mins = tf.argmin(diffs_to_poles, axis=1)
winner = tf.map_fn(lambda m: vectors[m, :], mins, dtype=tf.float64)

winners = sess.run(winner)

def vector_to_ratio(vector):
    primes = PRIMES[:vector.shape[0]]
    num = np.where(vector > 0, vector, np.zeros_like(primes))
    den = np.where(vector < 0, vector, np.zeros_like(primes))
    return (
        np.product(np.power(primes, num)), 
        np.product(primes ** np.abs(den))
    )

all_possible_pitches = set()

ratios = np.apply_along_axis(lambda row: vector_to_ratio(row), 1, winners)
np.apply_along_axis(lambda ratio: all_possible_pitches.add((ratio[0], ratio[1])), 1, ratios)

print(len(all_possible_pitches))
print(sorted(all_possible_pitches, key=lambda r: r[0] / r[1]))
