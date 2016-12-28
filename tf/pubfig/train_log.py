'''
PubFig face recognition sample
Author: Win Woo
'''

import tensorflow as tf
import os
import time
import math
import sys
import subprocess
import json
import tarfile
import logging

flags = tf.app.flags
FLAGS = flags.FLAGS

#===========================
# hyperparameters to use for training
flags.DEFINE_integer('train_batch_size', 100, 'Training batch size')
flags.DEFINE_integer('train_epochs', None, 'Training epochs')
flags.DEFINE_integer('valid_batch_size', 384, 'Validation batch size')
flags.DEFINE_integer('valid_epochs', None, 'Validation epochs')
flags.DEFINE_boolean('shuffle_batches', True, 'Whether to shuffle batches')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate')
flags.DEFINE_integer('num_classes', 48, 'Number of classification classes')
flags.DEFINE_float('keep_prob', 0.75, 'L2 dropout, percentage to keep with each training batch')
flags.DEFINE_integer('valid_steps', 100, 'Number of training steps between between validation steps')
flags.DEFINE_integer('image_summary_steps', 200, 'Number of training steps between generating an image summary')
flags.DEFINE_boolean('mod_input', True, 'Whether to perform random modification of images')
flags.DEFINE_integer('max_steps', 7000, 'Maximum number of training steps')

flags.DEFINE_integer('image_size', 96, 'Resize images to this width and height')
flags.DEFINE_integer('image_channels', 3, 'Number of image channels to use')
flags.DEFINE_integer('conv_size', 3, 'Size of convolution kernel (squared)')
flags.DEFINE_integer('wc1_layer_size', 16, 'Size of first layer (convolution)')
flags.DEFINE_integer('wc2_layer_size', 32, 'Size of second layer (convolution)')
flags.DEFINE_integer('wc3_layer_size', 64, 'Size of third layer (convolution)')
flags.DEFINE_integer('wc4_layer_size', 128, 'Size of fourth layer (convolution)')
flags.DEFINE_integer('wd1_layer_size', 512, 'Size of fifth layer (fully-connected)')
flags.DEFINE_integer('wd2_layer_size', 256, 'Size of sixth layer (fully-connected)')
flags.DEFINE_float('normal_stdev', 0.1, 'Standard deviation for normal weight distribution')

# grid_summary_x * grid_summary_y should equal wc1_layer_size
flags.DEFINE_integer('grid_summary_x', 4, 'Number of images on the x-axis of the summary grid')
flags.DEFINE_integer('grid_summary_y', 4, 'Number of images on the y-axis of the summary grid')

flags.DEFINE_string('train_file', '/tmp/data/train.txt', 'Path to training file')
flags.DEFINE_string('valid_file', '/tmp/data/valid.txt', 'Path to validation file')
flags.DEFINE_string('tmp_dir', '/tmp/', 'Temporary data path')
flags.DEFINE_string('gcs_export_uri', 'gs://wwoo-train/pubfig/export/', 'Path to output files')

flags.DEFINE_boolean('copy_from_gcs', True, 'Whether to copy files GCS')
flags.DEFINE_boolean('copy_to_gcs', True, 'Whether to copy logs to GCS')
flags.DEFINE_string('gcs_tarball_uri', 'gs://wwoo-train/pubfig/out.tar.gz', 'Path to GCS tarball source')
flags.DEFINE_string('data_path_prepend', '/tmp/', 'Path to prepend to training and validation files')

flags.DEFINE_boolean('save_model', True, 'Whether to save the model')
flags.DEFINE_float('accuracy_exit_threshold', 0.75, 'Exit training once this validation accuracy is reached')

# base path to prepend to training and validation data files
TENSOR_BASE_PATH = tf.constant(FLAGS.data_path_prepend)
#===========================

def get_image_label_list(image_label_file):
    filenames = []
    labels = []
    for line in open(image_label_file, "r"):
        filename, label = line[:-1].split('|')
        filenames.append(filename)
        labels.append(int(label))

    logging.info("get_image_label_list: read " + str(len(filenames)) \
        + " items")

    return filenames, labels

def read_image_from_disk(input_queue):
    label = input_queue[1]
    file_name = tf.reduce_join([TENSOR_BASE_PATH, input_queue[0]], 0)
    file_contents = tf.read_file(file_name)
    rgb_image = tf.image.decode_jpeg(file_contents, channels=FLAGS.image_channels,
        name="decode_jpeg")
    rgb_image  = tf.image.convert_image_dtype(rgb_image, dtype=tf.float32,
        name="image_convert_float32")
    rgb_image = tf.image.resize_images(rgb_image,
        [FLAGS.image_size, FLAGS.image_size])

    return rgb_image, label

def batch_inputs(train_file, batch_size=FLAGS.train_batch_size,
    num_epochs=FLAGS.train_epochs):

    image_list, label_list = get_image_label_list(train_file)
    input_queue = tf.train.slice_input_producer([image_list, label_list],
        num_epochs=num_epochs, shuffle=FLAGS.shuffle_batches)
    image, label = read_image_from_disk(input_queue)
    image = tf.reshape(image, [FLAGS.image_size, FLAGS.image_size, FLAGS.image_channels])

    if FLAGS.mod_input:
        # Do some random transformations to the images
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.5)
        image = tf.image.random_contrast(image, lower=0.2, upper=2.0)
        image = tf.image.random_hue(image, max_delta=0.08)
        image = tf.image.random_saturation(image, lower=0.2, upper=2.0)

    image_batch, label_batch = tf.train.batch([image, label],
        batch_size=batch_size)

    return image_batch, tf.one_hot(tf.to_int64(label_batch),
        FLAGS.num_classes, on_value=1.0, off_value=0.0)

def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1],
        padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)

def maxpool2d(x, k=2, layer=""):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1],
        strides=[1, k, k, 1], padding='SAME')

def conv_net(x, weights, biases, image_size, keep_prob=FLAGS.keep_prob):
    # Convolution and max pooling layers
    # Each max pooling layer reduces dimensionality by 2

    with tf.name_scope('conv_pool_1'):
        # Convolution and max pooling layer 1
        conv1 = conv2d(x, weights['wc1'], biases['bc1'])
        pool1 = maxpool2d(conv1, k=2)

    with tf.name_scope('conv_pool_2'):
        # Convolution and max pooling layer 2
        conv2 = conv2d(pool1, weights['wc2'], biases['bc2'])
        pool2 = maxpool2d(conv2, k=2)

    with tf.name_scope('conv_pool_3'):
        # Convolution and max pooling layer 3
        conv3 = conv2d(pool2, weights['wc3'], biases['bc3'])
        pool3 = maxpool2d(conv3, k=2)

    with tf.name_scope('conv_pool_4'):
        # Convolution and max pooling layer 4
        conv4 = conv2d(pool3, weights['wc4'], biases['bc4'])
        pool4 = maxpool2d(conv4, k=2)

    with tf.name_scope('fully_connected_1'):
        # Fully-connected layer
        fc1 = tf.reshape(pool4, [-1, weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope('fully_connected_2'):
        # Fully-connected layer
        fc2 = tf.add(tf.matmul(fc1, weights['wd2']), biases['bd2'])
        fc2 = tf.nn.relu(fc2)

        # Apply dropout
        fc2 = tf.nn.dropout(fc2, keep_prob)

    with tf.name_scope('output'):
        # Output, class prediction
        out = tf.add(tf.matmul(fc2, weights['out']), biases['out'])

    return out

def generate_image_summary(x, weights, biases, step, image_size=FLAGS.image_size):
    with tf.name_scope('generate_image_summary'):
        x =  tf.slice(x, [0, 0, 0, 0],
            [FLAGS.valid_batch_size, image_size, image_size, FLAGS.image_channels])
        x = tf.nn.conv2d(x, weights['wc1'], strides=[1, 1, 1, 1],
            padding='SAME')

        # Nifty grid image summary via:
        # http://stackoverflow.com/questions/33802336/visualizing-output-of-convolutional-layer-in-tensorflow

        x = tf.slice(x, [0, 0, 0, 0], [1, -1, -1, -1])
        x = tf.reshape(x, [image_size, image_size, FLAGS.wc1_layer_size])

        pad_xy = image_size + 4
        x = tf.image.resize_image_with_crop_or_pad(x, pad_xy, pad_xy)
        x = tf.reshape(x, [pad_xy, pad_xy, FLAGS.grid_summary_x, FLAGS.grid_summary_y])
        x = tf.transpose(x, [2, 0, 3, 1])
        x = tf.reshape(x, [1, pad_xy * FLAGS.grid_summary_x, pad_xy * FLAGS.grid_summary_y, 1])

        conv_summary = tf.summary.image("img_conv_{:05d}".format(step), x)
        relu_summary = tf.summary.image("img_relu_{:05d}".format(step), tf.nn.relu(x))

    return conv_summary, relu_summary

def gcs_copy(source, dest):
    logging.info('Recursively copying from %s to %s' %
        (source, dest))
    subprocess.check_call(['gsutil', '-q', '-m', 'cp', '-R']
        + [source] + [dest])

def extract_tarball(tarball, dest):
    tar = tarfile.open(tarball, 'r')
    tar.extractall(FLAGS.tmp_dir)

def main(argv=None):

    try:
        # Create tmp dir
        if not os.path.exists(FLAGS.tmp_dir):
            os.makedirs(FLAGS.tmp_dir)

        # Create directory for logs
        base_log_dir = os.path.join(FLAGS.tmp_dir, 'logs')
        if not os.path.exists(base_log_dir):
            os.makedirs(base_log_dir)

    except Exception, e:
        logging.info("Failed to create directory %s" % (base_log_dir))
        exit(1)

    try:
        # Copy training data
        if FLAGS.copy_from_gcs:
            gcs_copy(FLAGS.gcs_tarball_uri, FLAGS.tmp_dir)
            extract_tarball(os.path.join(FLAGS.tmp_dir, os.path.basename(FLAGS.gcs_tarball_uri)), 
                FLAGS.tmp_dir)

    except Exception, e:
        logging.info("Failed to download and extract tarball from %s to %s" % (FLAGS.gcs_tarball_uri, FLAGS.tmp_dir))
        exit(1) 

    # Read inventory of training images and labels
    with tf.name_scope('batch_inputs'):
        train_image_batch, train_label_batch = batch_inputs(FLAGS.train_file,
            batch_size=FLAGS.train_batch_size, num_epochs=FLAGS.train_epochs)
        valid_image_batch, valid_label_batch = batch_inputs(FLAGS.valid_file,
            batch_size=FLAGS.valid_batch_size, num_epochs=FLAGS.valid_epochs)

    # These are image and label batch placeholders which we'll feed in during training
    x_ = tf.placeholder("float32", shape=[None, FLAGS.image_size, FLAGS.image_size,
        FLAGS.image_channels])
    y_ = tf.placeholder("float32", shape=[None, FLAGS.num_classes])
    k_ = tf.placeholder("int64", shape=[None,])

    # Define tensor inputs to use during prediction
    inputs = {'key': k_.name, 'image': x_.name}
    tf.add_to_collection('inputs', json.dumps(inputs))

    # k is the image size after 4 maxpools
    k = int(math.ceil(FLAGS.image_size / 2.0 / 2.0 / 2.0 / 2.0))

    # Store weights for our convolution & fully-connected layers
    with tf.name_scope('weights'):
        weights = {
            'wc1': tf.Variable(tf.truncated_normal(
                [FLAGS.conv_size, FLAGS.conv_size, 1 * FLAGS.image_channels,
                    FLAGS.wc1_layer_size], stddev=FLAGS.normal_stdev)),
            'wc2': tf.Variable(tf.truncated_normal(
                [FLAGS.conv_size, FLAGS.conv_size, FLAGS.wc1_layer_size,
                    FLAGS.wc2_layer_size], stddev=FLAGS.normal_stdev)),
            'wc3': tf.Variable(tf.truncated_normal(
                [FLAGS.conv_size, FLAGS.conv_size, FLAGS.wc2_layer_size,
                    FLAGS.wc3_layer_size], stddev=FLAGS.normal_stdev)),
            'wc4': tf.Variable(tf.truncated_normal(
                [FLAGS.conv_size, FLAGS.conv_size, FLAGS.wc3_layer_size,
                    FLAGS.wc4_layer_size], stddev=FLAGS.normal_stdev)),
            'wd1': tf.Variable(tf.truncated_normal([k * k * FLAGS.wc4_layer_size,
                FLAGS.wd1_layer_size], stddev=FLAGS.normal_stdev)),
            'wd2': tf.Variable(tf.truncated_normal([FLAGS.wd1_layer_size,
                FLAGS.wd2_layer_size], stddev=FLAGS.normal_stdev)),
            'out': tf.Variable(tf.truncated_normal([FLAGS.wd2_layer_size,
                FLAGS.num_classes], stddev=FLAGS.normal_stdev))
        }

    # Store biases for our convolution and fully-connected layers
    with tf.name_scope('biases'):
        biases = {
            'bc1': tf.Variable(tf.truncated_normal([FLAGS.wc1_layer_size],
                stddev=FLAGS.normal_stdev)),
            'bc2': tf.Variable(tf.truncated_normal([FLAGS.wc2_layer_size],
                stddev=FLAGS.normal_stdev)),
            'bc3': tf.Variable(tf.truncated_normal([FLAGS.wc3_layer_size],
                stddev=FLAGS.normal_stdev)),
            'bc4': tf.Variable(tf.truncated_normal([FLAGS.wc4_layer_size],
                stddev=FLAGS.normal_stdev)),
            'bd1': tf.Variable(tf.truncated_normal([FLAGS.wd1_layer_size],
                stddev=FLAGS.normal_stdev)),
            'bd2': tf.Variable(tf.truncated_normal([FLAGS.wd2_layer_size],
                stddev=FLAGS.normal_stdev)),
            'out': tf.Variable(tf.truncated_normal([FLAGS.num_classes],
                stddev=FLAGS.normal_stdev))
        }

    # Define dropout rate to prevent overfitting
    keep_prob = tf.placeholder(tf.float32)

    # Build our graph
    pred = conv_net(x_, weights, biases, FLAGS.image_size, keep_prob)

    # Calculate loss
    with tf.name_scope('cross_entropy'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y_))
        cost_summary = tf.summary.scalar("cost_summary", cost)

    # Run optimizer step
    with tf.name_scope('train'):
        opt = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads = opt.compute_gradients(cost)
        train_step = opt.apply_gradients(grads)

    # Write summaries for all variables
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)

    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/grads', grad)

    # Evaluate model accuracy
    with tf.name_scope('predict'):
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        accuracy_summary = tf.summary.scalar("accuracy_summary", accuracy)

    # return an identity tensor that mirrors the input keys
    k = tf.identity(k_)

    # Define tensor outputs for prediction
    outputs = {'key': k.name, 'prediction': correct_pred.name}
    tf.add_to_collection('outputs', json.dumps(outputs))

    sess = tf.Session()

    train_writer = tf.summary.FileWriter(os.path.join(base_log_dir, 'train_logs'),
        sess.graph)
    valid_writer = tf.summary.FileWriter(os.path.join(base_log_dir, 'valid_logs'),
        sess.graph)

    summary_op = tf.summary.merge_all()
    #init_op = tf.initialize_all_variables() # use this for tensorflow 0.11rc0
    init_op = tf.global_variables_initializer() # use this for tensorflow 0.12rc0

    # we need init_local_op step only on tensorflow 0.10rc due to a regression from 0.9
    # https://github.com/tensorflow/models/pull/297
    # init_local_op = tf.initialize_local_variables()

    saver = tf.train.Saver()

    step = 0

    with sess.as_default():
        sess.run(init_op)
        # sess.run(init_local_op) # we need this only with tensorflow 0.10rc
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        try:
            while not coord.should_stop():
                step += 1
                x, y = sess.run([train_image_batch, train_label_batch])
                train_step.run(feed_dict={keep_prob: FLAGS.keep_prob,
                    x_: x, y_: y})

                if step % FLAGS.image_summary_steps == 0:
                    # write image summaries to view in tensorboard
                    x, y = sess.run([valid_image_batch, valid_label_batch])
                    conv_summary, relu_summary = generate_image_summary(x_, weights,
                        biases, step, FLAGS.image_size)
                    result = sess.run([conv_summary, relu_summary],
                        feed_dict={keep_prob: 1.0, x_: x, y_: y})
                    valid_writer.add_summary(result[0])
                    valid_writer.add_summary(result[1])

                if step % FLAGS.valid_steps == 0:
                    # training accuracy, loss and write summaries
                    result = sess.run([summary_op, accuracy, cost],
                        feed_dict={keep_prob: 1.0, x_: x, y_: y})
                    train_writer.add_summary(result[0], step)

                    logging.info("Step [%s] (train): accuracy: %s, loss: %s" %
                        (step, result[1], result[2]))

                    # validation accuracy, loss and write summaries
                    x, y = sess.run([valid_image_batch, valid_label_batch])
                    result = sess.run([summary_op, accuracy, cost],
                        feed_dict={keep_prob: 1.0, x_: x, y_: y})
                    valid_writer.add_summary(result[0], step)

                    logging.info("Step [%s] (valid): accuracy: %s, loss: %s" %
                        (step, result[1], result[2]))

                    if result[1] >= FLAGS.accuracy_exit_threshold or step >= FLAGS.max_steps:
                        logging.info("Step [%s] (complete)" % (step))
                        base_model_dir = os.path.join(FLAGS.tmp_dir, 'model')

                        if FLAGS.save_model:
                            if not os.path.exists(base_model_dir):
                                os.makedirs(base_model_dir)
                            save_path = saver.save(sess, os.path.join(base_model_dir,
                                'export'))

                        if FLAGS.copy_to_gcs:
                            gcs_copy(base_log_dir, FLAGS.gcs_export_uri)
                            gcs_copy(base_model_dir, FLAGS.gcs_export_uri)

                        # exit if the validation accuracy threshold is reached
                        break

        except tf.errors.OutOfRangeError:
            x, y = sess.run([valid_image_batch, valid_label_batch])
            result = sess.run([accuracy], feed_dict={keep_prob: 1.0,
                x_: x, y_: y})
            logging.info("Validation accuracy: %s" % result[0])

        finally:
            coord.request_stop()
            coord.join(threads)
            sess.close()

    return 0

if __name__ == '__main__':
    tf.app.run()

