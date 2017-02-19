import sys
import os
import progressbar as pb
import numpy as np
import random
from itertools import product
import mtp
from mtp import get_am_filename
import tflearn
import copy
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import l2_normalize
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

sys.path.append('./FlyLIB/')
import neuron

DIRECTORY_AMS = "/home/toosyou/ext/neuron_data/resampled_111_slow/"
SIZE_BATCH = 20
SIZE_VALIDATION = 50

SIZE_INPUT_DATA = [200, 200, 21]
SIZE_INPUT_RESIZE = SIZE_INPUT_DATA + [200]
SIZE_OUTPUT_DATA = [100, 100]

index_block_postive_sofar = 0
index_block_negtive_sofar = 0

def build_cnn_model():
    network_input_size = [None] + SIZE_INPUT_DATA # None 200 200 21

    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()

    # build CNN
    network = input_data(shape=network_input_size,
                         data_preprocessing=img_prep) # None 200 200 21
    network = conv_2d(network, 32, 3, activation='relu')
    network = max_pool_2d(network, 2) # 32 100 100 21
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2) # 64 50 50 21
    network = conv_2d(network, 64, 3, activation='relu')
    network = conv_2d(network, 64, 3, activation='relu')
    network = max_pool_2d(network, 2) # 64 25 25 21
    network = conv_2d(network, 64, 1, activation='relu') # 64 25 25 1
    network = fully_connected(network, 2048)
    network = fully_connected(network, np.prod(SIZE_OUTPUT_DATA)) # 10000
    network = regression(network, optimizer='adam',
                            loss='binary_crossentropy')
    model = tflearn.DNN(network)
    return model, network

def neuron_resize_test():
    train_mtp = mtp.MTP('train.mtp')
    for i in range(10):
        original_neuron_name = train_mtp[i].name
        neuron_name = DIRECTORY_AMS + get_am_filename(original_neuron_name)
        print(neuron_name)
        try:
            test_neuron = neuron.NeuronRaw(neuron_name)
        except:
            continue
        if test_neuron.valid == False:
            continue

        # copy for tips point
        test_tips_vol = copy.deepcopy(test_neuron)
        test_tips_vol.clear_intensity()
        test_tips_vol.read_from_points(train_mtp[i].coordinates, exaggerate=True)

        # orig
        print(test_neuron.size)
        print(test_neuron.intensity.shape)
        test_neuron.write_am(original_neuron_name+'.am')
        test_tips_vol.write_am(original_neuron_name+'_target.am')

        # resized
        test_neuron = test_neuron.resize([200, 200, 200])
        test_tips_vol = test_tips_vol.resize([200, 200, 200])
        print(test_neuron.size)
        print(test_neuron.intensity.shape)
        test_neuron.write_am(original_neuron_name+'_resized.am')
        test_tips_vol.write_am(original_neuron_name+'_target_resized.am')
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
    #validation_X, validation_Y = get_data(test_mtp, 0, SIZE_VALIDATION)
    validation_X, validation_Y, validation_index = test_mtp.get_sliced_data(0, SIZE_VALIDATION, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA)

    print("Accuracy begin: ", model.evaluate(validation_X, validation_Y))

    return

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
    main_train()
