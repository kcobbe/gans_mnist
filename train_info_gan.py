import math
import time
import numpy as np
import tensorflow as tf
from models import Model
import argparse

from plot_utils import Plotter

import utils

BATCH_SIZE = 100
NUM_EPOCHS = 300
Z_SIZE = 50
DISPLAY_INTERVAL = 100

NUM_CLASSES = 10

VISUALIZATION_LEN = 10

parser = argparse.ArgumentParser(description='Learn unsupervised labels on MNIST')
parser.add_argument("-l", "--labels", type=int, default=10, help="number of unsupervised labels to learn")

def sigmoid_loss(logits, targets):
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=targets)
    return tf.reduce_mean(loss)

def calc_error(num_unsupervised_labels, enc_labels, actual_labels):
    total_err = 0

    for cl in range(num_unsupervised_labels):
        actuals = actual_labels[np.where(enc_labels == cl)]
        if len(actuals) > 0:
            label_counts = np.bincount(actuals)
            chosen_label = np.argmax(label_counts)
            err_count = np.sum(label_counts) - label_counts[chosen_label]
            total_err += err_count

    cl_err = total_err / len(actual_labels)

    return cl_err

def create_label_data(num_unsupervised_labels):
    label_data = np.zeros((BATCH_SIZE, num_unsupervised_labels))
    unsupervised_labels = np.random.randint(num_unsupervised_labels, size=BATCH_SIZE)
    label_data[range(BATCH_SIZE),unsupervised_labels] = 1

    return label_data, unsupervised_labels

def main():
    args = parser.parse_args()
    num_unsupervised_labels = args.labels

    plotter = Plotter(rows=num_unsupervised_labels, columns=VISUALIZATION_LEN)

    train_data, train_labels, test_data, test_labels = utils.load_MNIST()

    input_shape = np.shape(train_data)
    input_dim = input_shape[1]
    c_dim = input_shape[3]

    model = Model(c_dim=c_dim, output_dim=input_dim, batch_size=BATCH_SIZE, z_dim=Z_SIZE)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    sample = tf.placeholder(tf.float32, shape=[BATCH_SIZE, Z_SIZE])
    batch_images = tf.placeholder(tf.float32, shape=[BATCH_SIZE, input_dim, input_dim, c_dim])
    aux_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, num_unsupervised_labels])

    generated_images = model.generator(sample, y=aux_data)
    D_fake, recon_encoded = model.discriminator(generated_images, y=None, num_out=[1, num_unsupervised_labels])
    D_real, input_encoded = model.discriminator(batch_images, y=None, num_out=[1, num_unsupervised_labels])

    loss_CL = tf.nn.softmax_cross_entropy_with_logits(labels=aux_data, logits=recon_encoded)
    loss_CL = tf.reduce_mean(loss_CL)

    loss_G = sigmoid_loss(D_fake, tf.ones_like(D_fake))
    loss_D = sigmoid_loss(D_fake, tf.zeros_like(D_fake)) + sigmoid_loss(D_real, tf.ones_like(D_real))

    g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

    opt_G = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(loss_G + loss_CL, var_list=g_vars)
    opt_D = tf.train.AdamOptimizer(2e-4, beta1=0.5).minimize(loss_D + loss_CL, var_list=d_vars)

    opts = [opt_D, opt_G]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    num_images = len(train_data)
    n_iterations = num_images // BATCH_SIZE

    start = time.time()

    for epoch in range(sess.run(global_step), NUM_EPOCHS):
        print('__EPOCH_{0}__'.format(epoch))
        perm = np.random.permutation(num_images)

        shuffled_images = train_data[perm]
        shuffled_labels = train_labels[perm]

        enc_labels = np.zeros_like(shuffled_labels)

        fake_accs = []
        real_accs = []
        losses = []

        for i in range(n_iterations):
            offset = (i * BATCH_SIZE) % num_images
            _batch_images = shuffled_images[offset:(offset + BATCH_SIZE)]
            _batch_labels = shuffled_labels[offset:(offset + BATCH_SIZE)]

            vec = utils.sample_code(BATCH_SIZE, Z_SIZE)
            _aux_data, _ = create_label_data(num_unsupervised_labels)

            feed_dict = {batch_images: _batch_images, sample: vec, aux_data: _aux_data}

            _, _d_reals, _d_fakes, _input_encoded, _losses = sess.run([opts, D_real, D_fake, input_encoded, [loss_G, loss_D, loss_CL]], feed_dict=feed_dict)

            enc_labels[offset:(offset + BATCH_SIZE)] = np.argmax(_input_encoded, axis=1)
            losses.append(_losses)

            fake_acc = np.sum(list(map(lambda x: 1 if x < .5 else 0, _d_fakes))) / BATCH_SIZE
            real_acc = np.sum(list(map(lambda x: 1 if x > .5 else 0, _d_reals))) / BATCH_SIZE

            fake_accs.append(fake_acc)
            real_accs.append(real_acc)

            if i % DISPLAY_INTERVAL == 0:
                print('Batch {0}/{1}'.format(i, n_iterations))
                print('  (Batch losses) Desc: {0:<8.3g} Gen: {1:<8.3g} Class: {2:<8.3g}'.format(*_losses))

        num_test = len(test_data)
        n_test_iterations = num_test // BATCH_SIZE
        enc_test_labels = np.zeros(num_test)

        vis_images = np.zeros((num_unsupervised_labels,VISUALIZATION_LEN,input_dim,input_dim,1))
        counts = np.zeros(num_unsupervised_labels, dtype=np.int32)

        for i in range(n_test_iterations):
            offset = (i * BATCH_SIZE) % num_test
            _batch_test = test_data[offset:(offset + BATCH_SIZE)]

            vec = utils.sample_code(BATCH_SIZE, Z_SIZE)
            _aux_data, unsupervised_labels = create_label_data(num_unsupervised_labels)
            feed_dict = {batch_images: _batch_test, sample: vec, aux_data: _aux_data}

            _test_encoded, _generated_images = sess.run([input_encoded, generated_images], feed_dict=feed_dict)

            for j, label in enumerate(unsupervised_labels):
                if counts[label] < VISUALIZATION_LEN:
                    vis_images[label,counts[label],:,:,:] = _generated_images[j]
                    counts[label] += 1

            enc_test_labels[offset:(offset + BATCH_SIZE)] = np.argmax(_test_encoded, axis=1)

        vis_images = np.reshape(vis_images, (num_unsupervised_labels*VISUALIZATION_LEN,input_dim,input_dim,1))
        plotter.plot(vis_images)

        cl_err = calc_error(num_unsupervised_labels, enc_labels, shuffled_labels)
        cl_test_err = calc_error(num_unsupervised_labels, enc_test_labels, test_labels)

        mean_losses = np.mean(losses, axis=0)

        now = time.time()
        elapsed = now - start

        print('Epoch Summary:')
        print('  (Mean losses) Desc: {0:<8.3g} Gen: {1:<8.3g} Class: {2:<8.3g}'.format(*mean_losses))
        print('  Discriminator mean fake acc:', np.mean(fake_accs))
        print('  Discriminator mean real acc:', np.mean(real_accs))
        print('  Train err:', cl_err)
        print('  Test err:', cl_test_err)
        print('  Elapsed (since start):', elapsed)

if __name__ == "__main__":
    main()
