import sys
import os
from os import mkdir, makedirs
import progressbar as pb
import numpy as np
import random
from itertools import product
import mtp
from mtp import get_am_filename
import tflearn
import copy
from scipy.misc import imsave
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import l2_normalize
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import multiprocessing
from multiprocessing import Process, Queue

sys.path.append('./FlyLIB/')
import neuron

DIRECTORY_AMS = "/home/toosyou/ext/neuron_data/resampled_111_slow/"
DIRECTORY_MODELS = '/home/toosyou/ext/models/'
DIRECTORY_VALIDS = '/home/toosyou/ext/valids/'
NUMBER_WORKERS = 5

SIZE_BATCH = 1
SIZE_VALIDATION = 1

SIZE_INPUT_DATA = [200, 200, 21]
SIZE_INPUT_RESIZE = SIZE_INPUT_DATA + [200]
SIZE_OUTPUT_DATA = [100, 100]

index_block_postive_sofar = 0
index_block_negtive_sofar = 0

def build_cnn_model():
    network_input_size = [None] + SIZE_INPUT_DATA # None 200 200 21

    img_prep = ImagePreprocessing()
    img_prep.add_samplewise_zero_center()
    img_prep.add_samplewise_stdnorm()

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
        test_tips_vol.clean_intensity()
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

def mtp_get_sliced_data(input_mtp, index_start, size, size_input, size_output, return_queue, index_worker = 0):
    result = list(input_mtp.get_sliced_data(index_start, size, size_input, size_output) )
    result.expend(index_worker)
    return_queue.put( result )
    return

def save_valids(model, index, validation_X, validation_Y):
    model_outputs = model.predict(validation_X)
    for i, output in enumerate(model_outputs):
        imsave(DIRECTORY_VALIDS+str(index)+'_'+str(i)+'_input.png', validation_X[i][:,:,10])
        imsave(DIRECTORY_VALIDS+str(index)+'_'+str(i)+'_target.png', validation_Y[i].reshape(SIZE_OUTPUT_DATA))
        imsave(DIRECTORY_VALIDS+str(index)+'_'+str(i)+'_output.png', np.array(output).reshape(SIZE_OUTPUT_DATA))
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
    valid_queue = Queue()
    valid_worker = Process(target=mtp_get_sliced_data,
                            args=(test_mtp, 0, SIZE_VALIDATION, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA, valid_queue))
    valid_worker.start()

    # make training batch and train
    makedirs(DIRECTORY_MODELS, exist_ok=True)
    makedirs(DIRECTORY_VALIDS, exist_ok=True)
    index_fit = 0
    index_train = 0
    # create NUMBER_WORKERS processes to read first batch
    train_queue = Queue()
    train_reading_workers = list()
    for i in range(NUMBER_WORKERS):
        read_process = Process(target=mtp_get_sliced_data,
                             args=(train_mtp, index_train, SIZE_BATCH, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA, train_queue, i))
        index_train += SIZE_BATCH
        train_reading_workers.append(read_process)
        read_process.start()

    while True:
        print('index_fit:', index_fit)
        # get previous reading result
        training_batch_X, training_batch_Y, _, index_worker = train_queue.get()
        train_reading_workers[index_worker].join()
        # begin reading process
        read_process = Process(target=mtp_get_sliced_data,
                                 args=(train_mtp, index_train, SIZE_BATCH, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA, train_queue, index_worker))
        index_train += SIZE_BATCH
        train_reading_workers[index_worker] = read_process
        read_process.start()
        # do cnn thing
        model.fit(training_batch_X, training_batch_Y, n_epoch=50,
                    show_metric=True)
        if index_fit % 5 == 0:
            model.save(DIRECTORY_MODELS + str(index_fit) + '.tfm')
        # wait for valid to finish reading
        if valid_worker.is_alive == True:
            print('Wait for valid to finish reading!')
            validation_X, validation_Y, _, _ = valid_queue.get()
            valid_worker.join()
        # save valid
        save_valids(model, index_fit, validation_X, validation_Y)
        # increase index fit
        index_fit += 1

    # final join
    train_queue.get()
    read_process.join()
    return

if __name__ == '__main__':
    main_train()
