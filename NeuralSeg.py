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
DIRECTORY_TRAIN_OUTPUT = '/home/toosyou/ext/train_output/'
NUMBER_WORKERS = 5

SIZE_BATCH = 10
SIZE_VALIDATION = 1

SIZE_INPUT_DATA = [200, 200, 21]
SIZE_INPUT_RESIZE = SIZE_INPUT_DATA + [200]
SIZE_OUTPUT_DATA = [100, 100]

index_block_postive_sofar = 0
index_block_negtive_sofar = 0

def build_cnn_model():
    network_input_size = [None] + SIZE_INPUT_DATA # None 200 200 21

    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()

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
                            loss='mean_square')
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
    result.append(index_worker)
    return_queue.put( result )
    return

def save_valids(model, index, X, Y, size=-1, is_training_set=False):
    # predict 'size' X or all X
    if size != -1:
        model_outputs = model.predict(X)
    else:
        model_outputs = model.predict(X[0:size])
    # set currect directory to save outputs/valids
    if is_training_set:
        directory = DIRECTORY_TRAIN_OUTPUT
    else:
        directory = DIRECTORY_VALIDS
    # output to image files
    for i, output in enumerate(model_outputs):
        imsave(directory+str(index)+'_'+str(i)+'_input.png', X[i][:,:,10])
        imsave(directory+str(index)+'_'+str(i)+'_target.png', Y[i].reshape(SIZE_OUTPUT_DATA))
        imsave(directory+str(index)+'_'+str(i)+'_output.png', np.array(output).reshape(SIZE_OUTPUT_DATA))
    return

def main_train(to_continue=False, name_model='checkpoint', init_index_fit=0, init_index_train=0):
    # build convolutional neural network with tflearn
    model, network = build_cnn_model()
    print("Done building cnn model!")

    # continue training with init weights loaded
    if to_continue:
        model.load(DIRECTORY_MODELS+name_model)
        print('Done load model weight to continue training')

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
    makedirs(DIRECTORY_TRAIN_OUTPUT, exist_ok=True)
    if to_continue: # continue training with initial index
        index_fit = init_index_fit
        index_train = init_index_train
    else:
        index_fit = 0
        index_train = 0
    get_valid = False # check if valid data is gotten
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
        print('Get worker:', index_worker, 'Size:', len(training_batch_Y) )
        train_reading_workers[index_worker].join()
        # begin reading process
        read_process = Process(target=mtp_get_sliced_data,
                                 args=(train_mtp, index_train, SIZE_BATCH, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA, train_queue, index_worker))
        index_train += SIZE_BATCH
        train_reading_workers[index_worker] = read_process
        read_process.start()

        # wait for valid to finish reading
        if get_valid == False:
            get_valid = True
            validation_X, validation_Y, _, _ = valid_queue.get()
            if valid_worker.is_alive == True:
                valid_worker.join()
        # do cnn thing
        model.fit(training_batch_X, training_batch_Y, n_epoch=300,
                    validation_set=(validation_X, validation_Y),
                    show_metric=True)
        if index_fit % 5 == 0:
            model.save(DIRECTORY_MODELS + str(index_fit) + '.tfm')

        # save valid
        save_valids(model, index_fit, validation_X, validation_Y)
        # save 10 training predict
        save_valids(model, index_fit, training_batch_X, training_batch_Y, size=10, is_training_set=True)
        # increase index fit
        index_fit += 1

    # final join, WIP
    return

if __name__ == '__main__':
    main_train()
