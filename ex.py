import tensorflow as tf
import tensorflow_probability as tfp
import os
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.Variable(0.)
def g(x):
    return np.cos(x)
def f(x):

    return g(x)+x



loss_fn2 = lambda: f(x)
losses2 = tfp.math.minimize(loss_fn2,
                           num_steps=100,
                           optimizer=tf.optimizers.Adam(learning_rate=0.1))

# In TF2/eager mode, the optimization runs immediately.
print("optimized value is {} with loss {}".format(x, losses2[-1]))