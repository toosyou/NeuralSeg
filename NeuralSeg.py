import sys
import os
from os import mkdir, makedirs
import progressbar as pb
import numpy as np
import random
from itertools import product
import mtp
from mtp import get_am_filename
from mtp import mtp_data_generator
import copy
from sklearn.utils import shuffle
from scipy.misc import imsave
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, reshape
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import l2_normalize
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

sys.path.append('./FlyLIB/')
import neuron

DIRECTORY_AMS = "/home/toosyou/ext/neuron_data/resampled_111_slow/"
DIRECTORY_MODELS = '/home/toosyou/ext/models/'
DIRECTORY_VALIDS = '/home/toosyou/ext/valids/'
DIRECTORY_TRAIN_OUTPUT = '/home/toosyou/ext/train_output/'
NUMBER_WORKERS = 3

SIZE_BATCH = 12
SIZE_VALIDATION = 1
MAX_SAMPLE_VALIDATION = 24
MAX_SAMPLE_PER_TRAINING_NEURON = 32
MAX_SAMPLE_TRAINING = MAX_SAMPLE_PER_TRAINING_NEURON * SIZE_BATCH
N_EPOCH_BATCH = 5
SIZE_MINI_BATCH = 32
N_EPOCH_MINI_BATCH = 1

SIZE_RESIZE_INPUT_Z = 100
SIZE_INPUT_DATA = [100, 100, 21]
SIZE_RESIZE_INPUT = [ SIZE_INPUT_DATA[0], SIZE_INPUT_DATA[1], SIZE_RESIZE_INPUT_Z ]
SIZE_OUTPUT_DATA = [25, 25]

def build_cnn_model():
    network_input_size = [None] + SIZE_INPUT_DATA + [1] # None 100 100 21 1

    img_prep = ImagePreprocessing()
    #img_prep.add_featurewise_zero_center()
    #img_prep.add_featurewise_stdnorm()

    # build CNN
    network = input_data(shape=network_input_size,
                            data_preprocessing=img_prep) # None 100 100 21 1
    network = conv_3d(network, 32, 3, activation='prelu')
    network = max_pool_3d(network, 2, strides=2, padding='same') # None 32 50 50 11 1
    network = conv_3d(network, 32, 3, activation='prelu')
    network = max_pool_3d(network, 2, strides=2, padding='same') # None 32 25 25 6 1
    network = conv_3d(network, 32, 3, activation='prelu') # None 32 25 25 6 1
    network = max_pool_3d(network, 2, strides=[1, 1, 1, 2, 1], padding='same') # None 32 25 25 3 1
    network = conv_3d(network, 128, 5, activation='prelu')
    network = conv_3d(network, 128, 5, activation='prelu')
    # network = conv_3d(network, 128, 5, activation='prelu')
    network = conv_3d(network, 1, [1, 1, 3], activation='prelu', padding='valid')
    network = flatten(network)
    # network = fully_connected(network, np.prod(SIZE_OUTPUT_DATA), activation='prelu')
    network = regression(network, optimizer='adam', loss='binary_crossentropy')

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

def save_valids(model, index, X, Y, size=-1, is_training_set=False):
    # predict 'size' X or all X
    if size != -1:
        model_outputs = model.predict(X[0:size])
    else:
        model_outputs = model.predict(X)
    # set currect directory to save outputs/valids
    if is_training_set:
        directory = DIRECTORY_TRAIN_OUTPUT
    else:
        directory = DIRECTORY_VALIDS
    # output to image files
    for i, output in enumerate(model_outputs):
        imsave(directory+str(index)+'_'+str(i)+'_input.png', X[i][:,:,10,0])
        imsave(directory+str(index)+'_'+str(i)+'_target.png', Y[i].reshape(SIZE_OUTPUT_DATA))
        imsave(directory+str(index)+'_'+str(i)+'_output.png', np.array(output).reshape(SIZE_OUTPUT_DATA))
    return

def train_batch(model, X, Y, validation_X, validation_Y):
    size_training = len(Y)
    number_batch = int(size_training/SIZE_MINI_BATCH)
    for epoch in range(N_EPOCH_BATCH):
        shuffle(X, Y) # shuffle data every epoch
        for iteration in range(number_batch):
            index_start = iteration*SIZE_MINI_BATCH
            index_end = index_start + SIZE_MINI_BATCH
            # check index bound
            if index_end > size_training:
                break
            # split batch
            batch_X = X[ index_start: index_end ]
            batch_Y = Y[ index_start: index_end ]
            model.fit(batch_X, batch_Y, n_epoch=N_EPOCH_MINI_BATCH,
                        # batch_size=SIZE_MINI_BATCH,
                        validation_set=(validation_X, validation_Y),
                        show_metric=True)
    return

def main_train(to_continue=False, name_model='checkpoint', init_index_fit=0, init_index_train=0):

    # prepare data
    train_mtp = mtp.MTP('train.mtp')
    test_mtp = mtp.MTP('test.mtp')

    # init generator
    train_data_generator = mtp_data_generator(train_mtp, SIZE_BATCH, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA, SIZE_RESIZE_INPUT, MAX_SAMPLE_TRAINING)
    test_data_generator = mtp_data_generator(test_mtp, SIZE_VALIDATION, SIZE_INPUT_DATA, SIZE_OUTPUT_DATA, SIZE_RESIZE_INPUT)

    # start reading validation and training set
    print("Reading validation set:")
    test_data_generator.start(number_worker=1)
    print("Reading training set:")
    train_data_generator.start(number_worker=NUMBER_WORKERS)

    # build convolutional neural network with tflearn
    model, network = build_cnn_model()
    print("Done building cnn model!")

    # continue training with init weights loaded
    if to_continue:
        model.load(DIRECTORY_MODELS+name_model)
        print('Done load model weight to continue training')

    # create directory for model, valid, training output
    makedirs(DIRECTORY_MODELS, exist_ok=True)
    makedirs(DIRECTORY_VALIDS, exist_ok=True)
    makedirs(DIRECTORY_TRAIN_OUTPUT, exist_ok=True)
    index_fit = init_index_fit
    index_train = init_index_train

    validation_X, validation_Y = test_data_generator.get(do_next=False)
    # limit the size of validation
    # if len(validation_Y) > MAX_SAMPLE_VALIDATION:
    #    validation_X = validation_X[0:MAX_SAMPLE_VALIDATION]
    #    validation_Y = validation_Y[0:MAX_SAMPLE_VALIDATION]
    print('Validation data gotten!')

    # start training
    while True:
        print('index_fit:', index_fit)
        # get previous reading result
        training_batch_X, training_batch_Y = train_data_generator.get()
        print('Training batch data gotten!')
        print('shrinked shape:', training_batch_X.shape, training_batch_Y.shape)

        # sample validation
        shuffle(validation_X, validation_Y)
        sampled_validation_X = validation_X[0:MAX_SAMPLE_VALIDATION]
        sampled_validation_Y = validation_Y[0:MAX_SAMPLE_VALIDATION]

        # do cnn thing
        train_batch(model, training_batch_X, training_batch_Y, sampled_validation_X, sampled_validation_Y)
        if index_fit % 5 == 0:
            model.save(DIRECTORY_MODELS + str(index_fit) + '.tfm')

        # save valid
        save_valids(model, index_fit, validation_X, validation_Y, size=MAX_SAMPLE_VALIDATION)
        print('Validation output saved!')
        # save 10 training predict
        save_valids(model, index_fit, training_batch_X, training_batch_Y, size=10, is_training_set=True)
        print('Training output saved')
        # increase index fit
        index_fit += 1

    # final join, WIP
    return

if __name__ == '__main__':
    main_train()
