import sys
import os
import progressbar as pb
import numpy as np
import random
import mtp
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.normalization import l2_normalize
from tflearn.layers.estimator import regression

sys.path.append('./FlyLIB/')
import neuron

DIRECTORY_AMS = "/Volumes/toosyou_ext/neuron_data/resampled_111_slow/"
SIZE_BATCH = 5

def build_cnn_model():
    network = input_data(shape=[None, 16, 16, 16, 1])
    network = conv_3d(network, 32, 3, activation='relu', regularizer='L2')
    network = max_pool_3d(network, 2) # 8 x 8 x 8 x 32
    network = conv_3d(network, 64, 3, activation='relu', regularizer='L2')
    network = max_pool_3d(network, 2) # 4 x 4 x 4 x 64
    network = fully_connected(network, 128, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 256, activation='tanh')
    network = dropout(network, 0.8)
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', loss='binary_crossentropy')
    return network

def get_am_filename(neuron_name):
    index_underline = neuron_name.rindex('_')
    rtn = neuron_name[0:index_underline+1] + 'resampled_1_1_1_ascii.am'
    return rtn;

def get_data( in_mtp, index_start, size):
    X = list()
    Y = list()
    for i, points in enumerate(in_mtp[index_start: index_start+size]):
        am_name = get_am_filename(points.name)
        # read neural raw data
        original_directory = os.getcwd()
        os.chdir(DIRECTORY_AMS)
        raw = neuron.NeuronRaw(am_name)
        os.chdir(original_directory)
        if raw.valid == False: # cannot read from am_name
            print("*****ERROR READING: ", am_name, " *****", file=sys.stderr)
            continue

        print("Blocklizing ", i, " of ", size, " ",am_name, " :")
        # blocklize with step = (8, 8, 8)
        pbar_block = pb.ProgressBar()
        for x in pbar_block(range(-8, raw.size[0]+8, 8)):
            for y in range(-8, raw.size[1]+8, 8):
                for z in range(-8, raw.size[2]+8, 8):
                    raw_block = raw.block([x, y, z])
                    if not raw_block.is_empty():
                        raw_block = raw_block.normalize() # normalize to (-1, 1)
                        X.append( raw_block )
                        # check if any tips is in the center of raw block
                        find_tips = False
                        for coor in points.coordinates:
                            if raw_block.is_in_the_center(coor):
                                find_tips = True
                                break
                        if find_tips:
                            Y.append(1)
                        else:
                            Y.append(0)
        print("X size: ", len(X))
    return X, Y

if __name__ == '__main__':

    # build convolutional neural network with tflearn
    network = build_cnn_model()
    model = tflearn.DNN(network)
    print("Done building cnn model!")

    # prepare data
    train_mtp = mtp.MTP('train.mtp')
    test_mtp = mtp.MTP('test.mtp')

    # make validation data with first 100 neurals from test
    print("Reading validation :")
    validation_X, validation_Y = get_data(test_mtp, 0, SIZE_BATCH)

    # make training batch and train
    train_size = train_mtp.size()
    for i in range( int(train_size/SIZE_BATCH) ):
        print("Reading training batch ", i, " :")
        training_batch_X, training_batch_Y = get_data(train_mtp, i*SIZE_BATCH, SIZE_BATCH)
        model.fit(training_batch_X, training_batch_Y, n_epoch=20,
                    validation_set=(validation_X, validation_Y),
                    show_metric=True)
        model.save('model.w')
