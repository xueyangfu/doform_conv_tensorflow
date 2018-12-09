import numpy as np
import tensorflow as tf
from deform_con2v import deform_conv2d
tf.GraphKeys.VARIABLES = tf.GraphKeys.GLOBAL_VARIABLES
with tf.variable_scope("tests"):
    arr = np.zeros((8, 6, 4, 5))
    a = tf.constant(arr, dtype=tf.float32)
    deform_conv = deform_conv2d(a, output_channals=10, kernel_size=3, stride=1, trainable=True, name="test")
    result = deform_conv.deform_con2v()

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(result)