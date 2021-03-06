import json
import tensorflow as tf
import itertools
import numpy as np
from cartesian import * 

# from contourtools.helpers import combinatorial_contour, get_bases

from tensorflow.python import debug as tf_debug

def combinatorial_contour(vec): 
    combos = np.array(list(itertools.combinations(vec, 2)))
    return combos[:, 0] - combos[:, 1]

def get_bases(length):
    return np.apply_along_axis(combinatorial_contour, 1, np.eye(length))[1:] * -1

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

def calc_func_graph(log_pitches, vectors, c=0.1, bounds=None):
    """
    log_pitches: tf.Variable that is the input coordinates
    vectors: arlike (not tf)
    c: coefficient to control the width of the bell curve
    """
    n_pitches = log_pitches.shape[1]
    pitch_distances = vector_pitch_distances(vectors)
    pitch_distances = tf.expand_dims(pitch_distances, -1)
    bases = get_bases(log_pitches.shape[-1] + 1)
    combinatorial_log_pitches = tf.abs(tf.tensordot(log_pitches, bases, 1))
    tiled_ones = tf.ones_like(pitch_distances) * -1.0
    combos = tf.tensordot(tiled_ones, combinatorial_log_pitches[None, :, :], 1)
    diffs = tf.abs(tf.add(tf.expand_dims(pitch_distances, 1), combos))
    scales = tf.exp(-1.0 * (diffs**2 / (2.0 * c**2)))

    prime_slice = PRIME_TRANSFORM[:vectors.shape[-1]]
    indigestibilities = tf.tensordot(tf.abs(vectors), prime_slice, 1)
    indigestibilities = tf.where(tf.not_equal(indigestibilities, tf.zeros_like(indigestibilities)), x=indigestibilities, y=tf.ones_like(indigestibilities))
    indigestibilities = indigestibilities[:, None, None] * tf.reciprocal(scales)
    
    if bounds is not None:
        if bounds.shape[0] != n_pitches:
            raise tf.errors.InvalidArgumentError(None, None, "bounds outer shape must == the number of log_pitches")
        pitch_diffs = tf.abs(pitch_distances)
        is_out_of_bounds = tf.logical_or(pitch_diffs < bounds[:, 0], pitch_diffs > bounds[:, 1])
        is_out_of_bounds = tf.tile(is_out_of_bounds[:, :, None], [1, 1, bases.shape[-1]])
        bases_mask = tf.equal(tf.abs(bases), 1.0)
        mask_me_again = tf.logical_not(tf.equal(tf.reduce_sum(bases, axis=0), 0.0))
        bases_mask = tf.logical_and(bases_mask, mask_me_again)
        is_relevant = tf.tile(bases_mask[None, :, :], [tf.shape(is_out_of_bounds)[0], 1, 1])
        is_both = tf.logical_and(is_out_of_bounds, is_relevant)
        is_both = tf.reduce_any(is_both, axis=1)
        is_both = tf.tile(is_both[:, None, :], [1, tf.shape(log_pitches)[0], 1])
        indigestibilities = tf.where(is_both, x=tf.fill(tf.shape(indigestibilities), tf.to_double(1.0e8)), y=indigestibilities)
    
    indigestibilities = tf.reduce_min(indigestibilities, axis=0)
    indigestibilities = tf.reduce_sum(indigestibilities, axis=-1)
    harmonicities = tf.to_double(bases.shape[-1]) * tf.reciprocal(indigestibilities)

    return harmonicities

def cost_func(log_pitches, vectors, c=0.1, name=None):
    return tf.subtract(tf.constant(1.0, dtype=tf.float64), calc_func_graph(log_pitches, vectors, c), name=name)

def vector_space_graph(n_primes, n_degrees, bounds=None, name=None):
    vectors = permutations(tf.range(-n_degrees, n_degrees+1, dtype=tf.float64), times=n_primes, name=name)
    if bounds is not None:
        pitch_distances = vector_pitch_distances(vectors)
        out_of_bounds_mask = tf.logical_and(tf.less_equal(pitch_distances, bounds[1]), tf.greater_equal(pitch_distances, bounds[0]))
        pitch_distances = tf.boolean_mask(pitch_distances, out_of_bounds_mask)
        vectors = tf.boolean_mask(vectors, out_of_bounds_mask)
    return vectors

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

    vectors = vector_space_graph(5, 4, name="vectors")

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

    step = 0.1
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
