import numpy as np
import tensorflow as tf

class deform_conv2d:

    def __init__(self, x, output_channals, kernel_size, stride, trainable, name):
        self.x = x
        self.output_channals = output_channals
        self.kernel_size = kernel_size
        self.stride = stride
        self.trainable = trainable
        self.name = name
        self.N = kernel_size ** 2   # Number of kernel elements in a bin

    # Definition of the regular 2D convolutional
    def conv(self, x, output_channals, mode):
        kernel_size = self.kernel_size
        stride = self.stride

        if mode == 'offset':
            layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=stride, padding='SAME', kernel_initializer = tf.zeros_initializer(), bias_initializer = tf.zeros_initializer())
        if mode == 'weight':
            layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=stride, padding='SAME', bias_initializer = tf.zeros_initializer())
        if mode == 'feature':
            layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=kernel_size, padding='SAME', kernel_initializer = tf.constant_initializer(0.5), bias_initializer = tf.zeros_initializer())
        
        return layer_output

    # Create the pn [1, 1, 1, 2N]
    def get_pn(self, dtype):

        kernel_size = self.kernel_size

        pn_x, pn_y = np.meshgrid(range(-(kernel_size-1)//2, (kernel_size-1)//2+1), range(-(kernel_size-1)//2, (kernel_size-1)//2+1), indexing="ij")

        # The order is [x1, x2, ..., y1, y2, ...]
        pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))

        pn = np.reshape(pn, [1, 1, 1, 2 * self.N])

        # Change the dtype of pn
        pn = tf.constant(pn, dtype)

        return pn

    # Create the p0 [1, h, w, 2N]
    def get_p0(self, x_size, dtype):

        batch_size, h, w, C = x_size

        p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
        p0_x = p0_x.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)
        p0_y = p0_y.flatten().reshape(1, h, w, 1).repeat(self.N, axis=3)
        p0 = np.concatenate((p0_x, p0_y), axis=3)

        # Change the dtype of p0
        p0 = tf.constant(p0, dtype)

        return p0

    # Create the p0 [h, w, 2]
    def get_q(self, x_size, dtype):

        batch_size, h, w, c = x_size

        q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
        q_x = q_x.flatten().reshape(h, w, 1)
        q_y = q_y.flatten().reshape(h, w, 1)
        q = np.concatenate((q_x, q_y), axis=2)

        # Change the dtype of q
        q = tf.constant(q, dtype)

        return q

    def reshape_x_offset(self, x_offset, kernel_size):

        batch_size, h, w, N, C = x_offset.get_shape().as_list()

        # Get the new_shape
        new_shape = [batch_size, h, w * kernel_size, C]
        x_offset = [tf.reshape(x_offset[:, :, :, s:s+kernel_size, :], new_shape) for s in range(0, N, kernel_size)]
        x_offset = tf.concat(x_offset, axis=2)

        # Reshape to final shape [batch_size, h*kernel_size, w*kernel_size, C]
        x_offset = tf.reshape(x_offset, [batch_size, h * kernel_size, w * kernel_size, C])

        return x_offset

    def deform_con2v(self):
            x = self.x
            N = self.N
            kernel_size = self.kernel_size

            batch_size, h, w, C = x.get_shape().as_list()

            # offset with shape [batch_size, h, w, 2N]
            offset = self.conv(x, 2 * N, "offset")

            # delte_weight with shape [batch_size, h, w, N * C]
            delte_weight = self.conv(x, N * C, "weight")
            delte_weight = tf.sigmoid(delte_weight)

            # pn with shape [1, 1, 1, 2N]
            pn = self.get_pn(offset.dtype)

            # p0 with shape [1, h, w, 2N]
            p0 = self.get_p0([batch_size, h, w, C], offset.dtype)

            # p with shape [batch_size, h, w, 2N]
            p = pn + p0 + offset

            # Reshape p to [batch_size, h, w, 2N, 1, 1]
            p = tf.reshape(p, [batch_size, h, w, 2 * N, 1, 1])

            # q with shape [h, w, 2]
            q = self.get_q([batch_size, h, w, C], offset.dtype)

            # Bilinear interpolation kernel G ([batch_size, h, w, N, h, w])
            gx = tf.maximum(1 - tf.abs(p[:, :, :, :N, :, :] - q[:, :, 0]), 0)
            gy = tf.maximum(1 - tf.abs(p[:, :, :, N:, :, :] - q[:, :, 1]), 0)
            G = gx * gy

            # Reshape G to [batch_size, h*w*N, h*w]
            G = tf.reshape(G, [batch_size, h * w * N, h * w])

            # Reshape x to [batch_size, h*w, C]
            x = tf.reshape(x, [batch_size, h*w, C])

            # x_offset with shape [batch_size, h, w, N, C]
            x_offset = tf.reshape(tf.matmul(G, x), [batch_size, h, w, N, C])

            # Reshape x_offset to [batch_size, h*kernel_size, w*kernel_size, C]
            x_offset = self.reshape_x_offset(x_offset, kernel_size)

            # Reshape delte_weight to [batch_size, h*kernel_size, w*kernel_size, C]
            delte_weight = tf.reshape(delte_weight, [batch_size, h*kernel_size, w*kernel_size, C])

            y = x_offset * delte_weight

            # Get the output of the deformable convolutional layer
            layer_output = self.conv(y, self.output_channals, "feature")

            return layer_output


    