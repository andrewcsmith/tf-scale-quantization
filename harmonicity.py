import json
import tensorflow as tf
import itertools
import numpy as np
from cartesian import * 
from contourtools.helpers import combinatorial_contour, get_bases

from tensorflow.python import debug as tf_debug

def calc_prime_transform(primes):
    """
    Transforms the primes for use in vectorized indigestibility calculations.

    Factors out all values related to p_{r} in the Barlow formula, so that we
    can calculate the indigestibility of a given integer through a vectorized
    matrix multiplication, rather than a manual nested loop.

    Returns the transformed primes (for use in `comb_harmonicity`).

    Parameters
    ----------
    primes : array_like
        Array of primes to be transformed.
    """
    return 2.0 * (((primes - 1) ** 2) / primes)

PRIMES = np.array([2, 3, 5, 7, 11, 13, 17, 19, 23, 29])
PRIME_TRANSFORM = calc_prime_transform(PRIMES)

def harmonicity_graph(n_primes, constellation):
    """
    The harmonicity of all elements in a vector.

    Note that n_primes must be provided at graph-creation time, because we need
    to know the size of the tensors to construct the graph properly.

    n_primes : integer 
        Number of primes in use in the constellation

    constellation : Tensor 
        Tensor of shape (?, n_primes), where each row in the
        first dimension will have its harmonicity calculated
    """
    prime_slice = PRIME_TRANSFORM[:n_primes]
    num = tf.sign(tf.tensordot(constellation, prime_slice, 1))
    den = tf.tensordot(tf.abs(constellation), prime_slice, 1)
    ones = tf.ones_like(num)
    zeros = tf.zeros_like(num)
    num = tf.where(tf.not_equal(num, zeros), x=num, y=ones)
    den = tf.where(tf.not_equal(den, zeros), x=den, y=ones)
    return num / den

def comb_harmonicity_graph(bases, n_primes, diffs):
    """
    Given a set of bases, number of primes, and basis
    coefficients (diffs from the first element), calculate
    the combinatorial harmonicity of the entire constellation.
    """
    web = tf.tensordot(bases.T, diffs, 1)
    return tf.abs(harmonicity_graph(n_primes, web))

def vector_pitch_distances(vectors):
    """
    Calculate the pitch distance (log2 of ratio) of each
    vector, provided as a row of prime factor exponents.
    """
    prime_slice = PRIMES[:vectors.shape[-1]]
    float_ratio = tf.reduce_prod(tf.pow(tf.constant(prime_slice, dtype=tf.float64), vectors), axis=1)
    return tf.log(float_ratio) / tf.log(tf.constant([2.0], dtype=tf.float64))

def calc_func_graph(log_pitches, vectors, c=0.1):
    """
    log_pitches: tf.Variable that is the input coordinates
    vectors: arlike (not tf)
    c: coefficient to control the width of the bell curve
    """
    pitch_distances = vector_pitch_distances(vectors)
    bases = get_bases(log_pitches.shape[-1] + 1)
    combinatorial_log_pitches = tf.abs(tf.tensordot(log_pitches, bases, 1))
    pitch_distances = tf.expand_dims(pitch_distances, -1)
    combinatorial_log_pitches = tf.expand_dims(combinatorial_log_pitches, 0)
    tiled_ones = tf.ones_like(pitch_distances) * -1.0
    diffs = tf.abs(tf.add(pitch_distances, tf.tensordot(tiled_ones, combinatorial_log_pitches, 1)))
    scales = tf.exp(-1.0 * (diffs**2 / (2.0 * c**2)))
    harmonicities = tf.abs(harmonicity_graph(vectors.shape[-1], vectors))
    harmonicities = tf.expand_dims(harmonicities, -1)
    return tf.reduce_mean(tf.reduce_max(harmonicities * scales, 0), -1)

def cost_func(log_pitches, vectors, c=0.1, name=None):
    return tf.subtract(1.0, calc_func_graph(log_pitches, vectors, c), name=name)

if __name__ == "__main__":
    # Initialization options here are because Windows does not
    # (by default) allow the gpu to allocate more memory
    # throughout the TF run.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    hooks = [tf_debug.LocalCLIDebugHook()]
    # Assignment has to be within sess.run
    starting_pitches = np.array([ 0.0, 0.0 ])
    log_pitches = tf.Variable(starting_pitches, name='log_pitches')
    init = tf.variables_initializer(tf.global_variables(), name="init")
    # log_pitches = tf.get_variable("log_pitches", shape=starting_pitches.shape, dtype=tf.float64)
    # sess.run(log_pitches.assign(starting_pitches))

    n_primes = 5
    n_degrees = 4
    vectors = permutations(tf.range(-n_degrees, n_degrees+1, dtype=tf.float64), times=n_primes, name="vectors")

    # Optimizer, setting the desired learning rate
    opt = tf.train.GradientDescentOptimizer(learning_rate=1.0e-4)

    # Vectors to calculate the loss function, so that we know when to stop.
    loss = cost_func(log_pitches, vectors, c=0.05, name="loss")
    # Calling #minimize on an Optimizer returns a graph that runs a single calculate
    # and update step.
    opt_op = opt.minimize(loss, var_list=[log_pitches], name="minimize")
    # We want to compute the new gradient after updating.
    compute_grad = opt.compute_gradients(loss, var_list=[log_pitches])
    # Get the norm of the gradient to see whether we're at the peak.
    grad_norms = [tf.nn.l2_loss(g) for g, v in compute_grad]
    grad_norm = tf.add_n(grad_norms, name="grad_norm")

    steps = np.array([starting_pitches * 1200.0])

    definition = sess.graph_def
    directory = '../../tf-polysynth'
    tf.train.write_graph(definition, directory, 'model.pb', as_text=False)

    log_vectors = vector_pitch_distances(vectors)
    diffs_to_poles = tf.abs(
        tf.transpose(tf.reshape(
            tf.tile(log_vectors, log_pitches.shape), 
            [log_pitches.shape[0], -1]))
        - log_pitches)

    mins = tf.argmin(diffs_to_poles, axis=0)
    winner = tf.map_fn(lambda m: vectors[m, :], mins, dtype=tf.float64)

    sess.run(init)
    MAX_ITERS = 100000

    def vector_to_ratio(vector):
        primes = PRIMES[:vector.shape[0]]
        num = np.where(vector > 0, vector, np.zeros_like(primes))
        den = np.where(vector < 0, vector, np.zeros_like(primes))
        return (
            np.product(np.power(primes, num)), 
            np.product(primes ** np.abs(den))
        )

    def run_minimizer():
        for i in range(MAX_ITERS):
            _, norm, out_pitches = sess.run([opt_op, grad_norm, log_pitches])
            # steps = np.vstack([steps, out_pitches * 1200.0])
            if (norm < 1.0e-10):
                # print("Converged at iteration: ", i)
                # out = log_pitches.eval(session=sess)
                print(out_pitches * 1200.0)
                break

    all_possible_pitches = set()

    step = 0.01
    x = np.arange(0.0, 1.0, step)
    y = np.arange(0.0, 1.0, step)
    xv, yv = np.meshgrid(x, y, sparse=False)
    starting_coordinates = np.array([xv, yv]).reshape(2, -1).T

    for i in starting_coordinates:
        print(i)
        assign_pitches_op = log_pitches.assign(i, use_locking=True)
        sess.run(assign_pitches_op)
        run_minimizer()

        ms = sess.run([mins])
        print(ms)
        winning_vector = np.array(sess.run([winner]))[0, :]
        print(winning_vector)
        for v in winning_vector:
            ratio = vector_to_ratio(v)
            all_possible_pitches.add(ratio)
            print(ratio)

    print("All possible pitches:")
    print(len(all_possible_pitches))
    print(all_possible_pitches)

    # with open('frequencies.json', 'w') as f:
    #     f.write(json.dumps(list(map(lambda x: list(x), steps))))

    # print(steps)
