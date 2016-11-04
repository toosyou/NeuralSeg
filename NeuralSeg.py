import sys
import os
import progressbar as pb
import numpy as np
import random
import mtp
import tensorflow as tf

sys.path.append('./FlyLIB/')
import neuron

SIZE_TIPS = 27898
# SIZE_TIPS = 20
DIRECTORY_AMS = "/Volumes/toosyou_ext/neuron_data/resampled_111_slow/"


# get weight from truncated normal distribution
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# initialized bias with constant 0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, w):
    return tf.nn.conv3d(x, w, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')


if __name__ == '__main__':


    # build tensorflow model for 3d convolutional neural network for tips

    # create a session
    sess = tf.InteractiveSession()

    # input images and labes
    x = tf.placeholder(tf.float32, shape=[None, 16*16*16])
    y_ = tf.placeholder(tf.float32, shape=[None, 2]) # having tips or not

    # convolutional layer
    w_conv1 = weight_variable([3, 3, 3, 1, 32])
    b_conv1 = bias_variable([32])

    # reshape input to 5d tensor
    x_volume = tf.reshape(x, [-1, 16, 16, 16, 1])

    h_conv1 = tf.nn.relu(conv3d(x_volume, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2x2(h_conv1)

    # second convolutional layer
    w_conv2 = weight_variable([3, 3, 3, 32, 64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv3d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2x2(h_conv2)

    # densely connected layer
    w_fc1 = weight_variable([4 * 4 * 4 * 64, 1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2, [-1, 4 * 4 * 4 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # dropout to reduce overfitting
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # output layer
    w_fc2 = weight_variable([1024, 2])
    b_fc2 = bias_variable([2])

    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # train
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                                  tf.log(y_conv), reduction_indices=[1]))
    train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

    # evaluate
    correct_predicetion = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predicetion, tf.float32))

    # initial all variables
    sess.run(tf.initialize_all_variables())

    '''
    train_mtp = mtp.MTP('train.mtp')
    print(train_mtp[0].name)
    for coordinates in train_mtp[0].coordinates:
        print(coordinates)
    '''
    # neurons = list()


    # read am files
    '''
    original_directory = os.getcwd()
    os.chdir(DIRECTORY_AMS)
    pbar = pb.ProgressBar()
    for t in pbar(tips[0:1]):
        neuron_name = t.neuron_am_name[:-7] + 'resampled_1_1_1_ascii.am'
        tmp_neuron = neuron.NeuronRaw( neuron_name )
        if tmp_neuron.valid == True:
            neurons.append( tmp_neuron )

    os.chdir(original_directory)

    # test neuron.block()
    try:
        os.chdir('block_test')
    except:
        os.mkdir('block_test', 0o755)
        os.chdir('block_test')
    for x in range(0, neurons[0].size[0], 16):
        for y in range(0, neurons[0].size[1], 16):
            for z in range(0, neurons[0].size[1], 16):
                block = neurons[0].block(start_point=(x, y, z))
                if block.is_empty() == False:
                    block.write_am(str(x)+'_'+str(y)+'_'+str(z)+'.am')
    os.chdir(original_directory)
    '''
