import tensorflow as tf
import math

STRIDE_1 = [1, 1, 1, 1]
STRIDE_2 = [1, 2, 2, 1]
STRIDE_4 = [1, 4, 4, 1]

def weight_norm_layer(input_, output_size, scope="weight_norm", bias_start=0.0, train_scale=False):
    with tf.variable_scope(scope):
        num_inputs = input_.get_shape()[1].value
        w_mat = tf.get_variable("W", [num_inputs, output_size], tf.float32, tf.contrib.layers.xavier_initializer(uniform=False))
        bias = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(bias_start))
        w_scale = tf.get_variable('w_scale', [output_size], initializer=tf.constant_initializer(1.0), trainable=train_scale)
        w_mat = tf.multiply(w_mat, w_scale) / tf.sqrt(1e-6 + tf.reduce_sum(tf.square(w_mat),axis=0))

    return tf.matmul(input_, w_mat) + bias

def gaussian_noise_layer(input_layer, std, is_training):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32) 
    out = tf.cond(tf.equal(is_training, tf.constant(True)), lambda: input_layer + noise, lambda: input_layer)
    
    return out

def batch_norm(x):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)

def conv2d_layer(input_, output_dim, k_d=5, d_d=2, stddev=0.02, name="conv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_d, k_d, input_.get_shape()[-1], output_dim], initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_d, d_d, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())

    return conv

def deconv2d(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="deconv2d"):
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                  initializer=tf.random_normal_initializer(stddev=stddev))

        deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape, strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())

    return deconv

def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak*x)

def linear_layer(input_, output_size, scope="linear", stddev=0.02, bias_start=0.0):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope):
        matrix = tf.get_variable("W", [shape[1], output_size], tf.float32, tf.contrib.layers.xavier_initializer(uniform=False))
        bias = tf.get_variable("b", [output_size], initializer=tf.constant_initializer(bias_start))

    return tf.matmul(input_, matrix) + bias

class Model(object):
    def __init__(self, batch_size=64, output_dim=32, z_dim=100, c_dim=3):
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.c_dim = c_dim

    def discriminator(self, image, y=None, num_out=1, scope='discriminator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            k = 64

            h = lrelu(conv2d_layer(image, k, name='d_h0_conv'))
            h = lrelu(batch_norm(conv2d_layer(h, k*2, name='d_h1_conv')))

            h = tf.reshape(h, [self.batch_size, -1])

            if not y == None:
                h = tf.concat([h, y], axis=1)
                
            h = lrelu(batch_norm(linear_layer(h, 1024, 'd_final_lin')))

            if hasattr(num_out, "__len__"):
                outs = []

                for (i, n) in enumerate(num_out):
                    out = linear_layer(h, n, 'd_out' + str(i) + '_lin')
                    outs.append(out)

                return outs

            out = linear_layer(h, num_out, 'd_out1_lin')

            return out

    def generator(self, z, y=None, scope='generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            k = 64

            if not y == None:
                z = tf.concat([z, y], axis=1)

            def conv_out_size_same(size, stride):
                return int(math.ceil(float(size) / float(stride)))

            s_h, s_w = self.output_dim, self.output_dim
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)

            h = tf.nn.relu(batch_norm(linear_layer(z, k*2*s_h4*s_w4, 'g_h1_lin')))
            h = tf.reshape(h, [self.batch_size, s_h4, s_w4, k*2])
            h = tf.nn.relu(batch_norm(deconv2d(h, [self.batch_size, s_h2, s_w2, k*2], name='g_h2')))

            out = tf.nn.sigmoid(deconv2d(h, [self.batch_size, s_h, s_w, self.c_dim], name='g_h_out'))

            return out

    def generator_mlp(self, z, scope='generator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            h = z
            h = tf.nn.softplus(batch_norm(linear_layer(h, 500, 'l1')))
            h = tf.nn.softplus(batch_norm(linear_layer(h, 500, 'l2')))
            out = tf.nn.sigmoid(linear_layer(h, self.output_dim**2, 'l3'))

        return out

    def discriminator_mlp(self, image, is_training, num_out=1, scope='discriminator'):
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            k = 250

            h = image

            h = gaussian_noise_layer(h, 0.3, is_training=is_training)
            h = tf.nn.relu(weight_norm_layer(h, 4 * k, 'l1'))
            h = gaussian_noise_layer(h, 0.5, is_training=is_training)
            h = tf.nn.relu(weight_norm_layer(h, 2 * k, 'l2'))
            h = gaussian_noise_layer(h, 0.5, is_training=is_training)
            h = tf.nn.relu(weight_norm_layer(h, k, 'l3'))
            h = gaussian_noise_layer(h, 0.5, is_training=is_training)
            h = tf.nn.relu(weight_norm_layer(h, k, 'l4'))
            h = gaussian_noise_layer(h, 0.5, is_training=is_training)
            h = tf.nn.relu(weight_norm_layer(h, k, 'l5'))
            h_match = h
            h = gaussian_noise_layer(h, 0.5, is_training=is_training)
            out = weight_norm_layer(h, num_out, 'l_out', train_scale=True)

        return out, h_match