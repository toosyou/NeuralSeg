import sys
import os
import progressbar as pb
import numpy as np
import random
from itertools import product
import mtp
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.normalization import l2_normalize
from tflearn.layers.estimator import regression

sys.path.append('./FlyLIB/')
import neuron

DIRECTORY_AMS = "/home/toosyou/ext/neuron_data/resampled_111_slow/"
SIZE_BATCH = 20
SIZE_VALIDATION = 20

size_input_data = [200, 200, 200]

index_block_postive_sofar = 0
index_block_negtive_sofar = 0

def build_cnn_model():
    network = input_data(shape=[None].extend(size_input_data)) # 200 200 200
    network = conv_3d(network, 32, 3, activation='relu')
    network = max_pool_3d(network, 2) # 32 100 100 100
    network = conv_3d(network, 64, 3, activation='relu')
    network = conv_3d(network, 64, 3, activation='relu')
    network = max_pool_3d(network, 2) # 64 50 50 50
    network = conv_3d(network, 128, 3, activation='relu')
    network = conv_3d(network, 128, 3, activation='relu')
    network = max_pool_3d(network, 2) # 128 25 25 25
    network = conv_3d(network, 128, 1, activation='relu')
    network = fully_connected(network, 2048)
    network = fully_connected(network, np.prod(size_input_data))
    network = regression(network, optimizer='adam',
                            loss='binary_crossentropy')
    model = tflearn.DNN(network)
    return model, network

def get_am_filename(neuron_name):
    index_underline = neuron_name.rindex('_')
    rtn = neuron_name[0:index_underline+1] + 'resampled_1_1_1_ascii.am'
    return rtn;

def get_data( in_mtp, index_start, size, balance=False):
    global index_block_postive_sofar
    global index_block_negtive_sofar
    X = list()
    Y = list()
    number_tip_block = 0
    # change directory to ams
    original_directory = os.getcwd()
    os.chdir(DIRECTORY_AMS)
    for i, points in enumerate(in_mtp[index_start: index_start+size]):
        am_name = get_am_filename(points.name)
        # read neural raw data
        raw = neuron.NeuronRaw(am_name)
        if raw.valid == False: # cannot read from am_name
            print("*****ERROR READING: ", am_name, " *****", file=sys.stderr)
            continue

        print("Blocklizing ", i, " of ", size, " ",am_name, " :")
        # blocklize with step = (8, 8, 8)
        pbar_block = pb.ProgressBar()
        for x, y, z in pbar_block( product( range(-8, raw.size[0]+8, 8), range(-8, raw.size[1]+8, 8), range(-8, raw.size[2]+8, 8)) ):
            raw_block = raw.block([x, y, z])
            if not raw_block.is_empty():
                raw_block = raw_block.normalize(rg=[-1, 1])
                X.append( raw_block.intensity.reshape( (16, 16, 16, 1) ) ) # normalize to (-1, 1)
                # check if any tips is in the center of raw block
                if raw_block.points_in_the_center(points.coordinates):
                    Y.append(1)
                    raw_block.normalize(rg=[0, 1]).write_am('/home/toosyou/projects/NeuralSeg/test_am/positive/' + str(index_block_postive_sofar) + '.am')
                    index_block_postive_sofar += 1
                    # make more positive data using mirroring
                    for x, y, z in product( (0, 1), repeat=3 ):
                        if (x, y, z) != (0, 0, 0):
                            X.append( raw_block.mirror([x, y, z]).intensity.reshape(16, 16, 16, 1) )
                            Y.append(1)
                    number_tip_block += 8
                else:
                    raw_block.normalize(rg=[0, 1]).write_am('/home/toosyou/projects/NeuralSeg/test_am/negtive/' + str(index_block_negtive_sofar) + '.am')
                    index_block_negtive_sofar += 1
                    Y.append(0)
        print("X size: ", len(X))
        print("# of non-tip block: ", len(X) - number_tip_block)
        print("# of tip block: ", number_tip_block)

    X = np.array(X)
    Y = np.array(tflearn.data_utils.to_categorical(Y, 2) )
    # change directory back
    os.chdir(original_directory)
    return X, Y

def neuron_resize_test():
    train_mtp = mtp.MTP('train.mtp')
    for i in range(10):
        neuron_name = DIRECTORY_AMS + get_am_filename(train_mtp[i].name)
        print(neuron_name)
        try:
            test_neuron = neuron.NeuronRaw(neuron_name)
        except:
            continue
        if test_neuron.valid == False:
            continue
        # orig
        print(test_neuron.size)
        print(test_neuron.intensity.shape)
        test_neuron.write_am(str(i)+'_original.am')
        # resized
        test_neuron = test_neuron.resize([200, 200, 200])
        print(test_neuron.size)
        print(test_neuron.intensity.shape)
        test_neuron.write_am(str(i)+'_resized.am')
    return

def main_train():
    # build convolutional neural network with tflearn
    model, network = build_cnn_model()
    print("Done building cnn model!")

    # prepare data
    train_mtp = mtp.MTP('train.mtp')
    test_mtp = mtp.MTP('test.mtp')

    # make validation data with first 100 neurals from test
    print("Reading validation :")
    validation_X, validation_Y = get_data(test_mtp, 0, SIZE_VALIDATION)

    print("Accuracy begin: ", model.evaluate(validation_X, validation_Y))

    # make training batch and train
    train_size = train_mtp.size()
    for i in range( int(train_size/SIZE_BATCH) ):
        print("Reading training batch ", i, " :")
        training_batch_X, training_batch_Y = get_data(train_mtp, i*SIZE_BATCH, SIZE_BATCH)
        model.fit(training_batch_X, training_batch_Y, n_epoch=10,
                    validation_set=(validation_X, validation_Y),
                    show_metric=True)
        model.save('model.tfl')
        print("Accuracy: ", model.evaluate(validation_X, validation_Y))

    return

if __name__ == '__main__':
    neuron_resize_test()
