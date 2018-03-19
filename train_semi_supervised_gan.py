import math
import time
import numpy as np
import tensorflow as tf
from models import Model
import argparse

from plot_utils import Plotter

import utils


BATCH_SIZE = 100
Z_SIZE = 100
DISPLAY_INTERVAL = 100

NUM_CLASSES = 10
INPUT_DIM = 28

NUM_EPOCHS = 300
LEARNING_DECAY = 1

parser = argparse.ArgumentParser(description='Use semi-supervised GAN to learn MNIST labels')
parser.add_argument("-cl", "--class-labels", type=int, default=10, help="number of labels per class during training")

def classify(sess, args, batch_images_labeled, training, data, labels, is_training=False):
    cl_accs = []

    num_images = len(data)

    perm = np.random.permutation(num_images)

    shuffled_data = data[perm]
    if len(labels) > 0:
        shuffled_labels = labels[perm]

    for i in range(num_images // BATCH_SIZE):
        offset = (i * BATCH_SIZE) % num_images

        _batch_images = shuffled_data[offset:(offset + BATCH_SIZE)]

        feed_dict = {batch_images_labeled: _batch_images, training: is_training}

        [_d_real_lab] = sess.run(args, feed_dict=feed_dict)

        if len(labels) == 0:
            cl_acc = 0
        else:
            _batch_labels = shuffled_labels[offset:(offset + BATCH_SIZE)]
            cl_is_correct = list(map(lambda kv: 1 if np.argmax(kv[1]) == _batch_labels[kv[0]] else 0, enumerate(_d_real_lab)))
            cl_acc = np.sum(cl_is_correct) / BATCH_SIZE
        
        cl_accs.append(cl_acc)

    return np.mean(cl_accs)

def train(images, labels, test_data, test_labels, labeled_data, chosen_labels, plotter):
    model = Model(c_dim=1, output_dim=INPUT_DIM, batch_size=BATCH_SIZE, z_dim=Z_SIZE)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    sample = tf.placeholder(tf.float32, shape=[BATCH_SIZE, Z_SIZE])
    batch_images_labeled = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_DIM * INPUT_DIM])
    batch_labels = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_CLASSES], name='labels')
    batch_images_unlabeled = tf.placeholder(tf.float32, shape=[BATCH_SIZE, INPUT_DIM * INPUT_DIM])
    training = tf.placeholder(tf.bool, name='training')
    learn_rate = tf.placeholder(tf.float32)

    generated_images = model.generator_mlp(sample)

    num_out = NUM_CLASSES

    D_fake, fake_match = model.discriminator_mlp(generated_images, training, num_out=num_out)
    D_real_lab, _ = model.discriminator_mlp(batch_images_labeled, training, num_out=num_out)
    D_real_unlab, real_match_unlab = model.discriminator_mlp(batch_images_unlabeled, training, num_out=num_out)

    fake_match = tf.reduce_mean(fake_match, axis=0)
    real_match_unlab = tf.reduce_mean(real_match_unlab, axis=0)
    loss_G = tf.reduce_mean(tf.square(fake_match - real_match_unlab))

    class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=D_real_lab, labels=tf.argmax(batch_labels, axis=1))
    class_loss = tf.reduce_mean(class_loss)

    z_unlab = tf.reduce_sum(tf.exp(D_real_unlab), axis=1)
    z_fake = tf.reduce_sum(tf.exp(D_fake), axis=1)
    p_unlab = z_unlab / (z_unlab + 1)
    p_fake = z_fake / (z_fake + 1)

    real_unlab_loss = tf.losses.log_loss(labels=tf.ones_like(p_unlab), predictions=p_unlab)
    fake_loss = tf.losses.log_loss(labels=tf.zeros_like(p_fake), predictions=p_fake)

    loss_D = .5 * class_loss + real_unlab_loss + fake_loss

    g_vars = [var for var in tf.trainable_variables() if 'generator' in var.name]
    d_vars = [var for var in tf.trainable_variables() if 'discriminator' in var.name]

    opt_G = tf.train.AdamOptimizer(learn_rate, beta1=0.5).minimize(loss_G, var_list=g_vars)
    opt_D = tf.train.AdamOptimizer(learn_rate, beta1=0.5).minimize(loss_D, var_list=d_vars)

    opts = [opt_D, opt_G]

    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    num_images = len(images)
    num_labeled_images = len(labeled_data)
    n_iterations = num_images // BATCH_SIZE

    start = time.time()

    for epoch in range(sess.run(global_step), NUM_EPOCHS):
        print('__EPOCH_{0}__'.format(epoch))
        perm = np.random.permutation(num_images)

        _learn_rate = 3e-4 * (LEARNING_DECAY ** epoch)

        shuffled_images = images[perm]

        fake_accs = []
        real_accs = []

        for i in range(n_iterations):
            offset = (i * BATCH_SIZE) % len(images)
            _batch_images = shuffled_images[offset:(offset + BATCH_SIZE)]

            vec = utils.sample_code(BATCH_SIZE, Z_SIZE)
            feed_dict = {batch_images_unlabeled: _batch_images, sample: vec, training: True, learn_rate: _learn_rate}

            perm2 = np.random.permutation(num_labeled_images)
            feed_dict[batch_images_labeled] = labeled_data[perm2]
            feed_dict[batch_labels] = chosen_labels[perm2]

            _, _dl_real_unlab, _dl_fake = sess.run([opts, p_unlab, p_fake] , feed_dict=feed_dict)

            fake_acc = np.sum(list(map(lambda x: 1 if np.sum(x) < .5 else 0, _dl_fake))) / BATCH_SIZE
            real_acc = np.sum(list(map(lambda x: 1 if np.sum(x) > .5 else 0, _dl_real_unlab))) / BATCH_SIZE

            fake_accs.append(fake_acc)
            real_accs.append(real_acc)

            if i % DISPLAY_INTERVAL == 0:
                result = sess.run([generated_images, loss_G, loss_D], feed_dict=feed_dict)
                [_generated_images, err_G, err_D] = result

                print('Batch {0}/{1}'.format(i, n_iterations))

                _generated_images = np.reshape(_generated_images, (BATCH_SIZE, INPUT_DIM, INPUT_DIM, 1))
                plotter.plot(_generated_images)

        args = [D_real_lab]

        print('Calculating test acc...')
        test_acc = classify(sess, args, batch_images_labeled, training, test_data, test_labels, False)

        now = time.time()
        elapsed = now - start

        print('Epoch Summary:')
        print('  Learn rate:', _learn_rate)
        print('  Discriminator mean fake acc:', np.mean(fake_accs))
        print('  Discriminator mean real acc:', np.mean(real_accs))
        print('  Test accuracy:', test_acc)
        print('  Elapsed (since start):', elapsed)

def main():
    args = parser.parse_args()
    plotter = Plotter()

    labels_per_class = args.class_labels

    train_data, train_labels, test_data, test_labels = utils.load_MNIST()

    num_train = len(train_data)

    train_data = np.reshape(train_data, (num_train, INPUT_DIM * INPUT_DIM))
    test_data = np.reshape(test_data, (len(test_data), INPUT_DIM * INPUT_DIM))

    counts = np.zeros(NUM_CLASSES)

    chosen_size = labels_per_class * NUM_CLASSES

    labeled_data = np.zeros((chosen_size, INPUT_DIM * INPUT_DIM))
    chosen_labels = np.zeros((chosen_size, NUM_CLASSES))
    new_idx = 0

    for idx, label in enumerate(train_labels):
        if counts[label] < labels_per_class:
            counts[label] += 1
            labeled_data[new_idx] = train_data[idx]
            chosen_labels[new_idx][label] = 1
            new_idx += 1

    train(train_data, train_labels, test_data, test_labels, labeled_data, chosen_labels, plotter)

if __name__ == "__main__":
    main()
