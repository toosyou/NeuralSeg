import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, reshape
from tflearn.layers.conv import conv_3d, max_pool_3d
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import l2_normalize
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

def get_slicing_model(input_size, output_size):
    network_input_size = [None] + input_size + [1] # None 100 100 21 1

    img_prep = ImagePreprocessing()

    # build CNN
    network = input_data(shape=network_input_size,
                            data_preprocessing=img_prep) # None 100 100 21 1
    network = conv_3d(network, 32, 3, activation='relu')
    network = max_pool_3d(network, 2, strides=2, padding='same') # None 32 50 50 11 1
    network = conv_3d(network, 32, 3, activation='relu')
    network = conv_3d(network, 32, 3, activation='relu')
    network = max_pool_3d(network, 2, strides=2, padding='same') # None 32 25 25 6 1
    network = conv_3d(network, 32, 3, activation='relu')
    network = conv_3d(network, 32, 3, activation='relu') # None 32 25 25 6 1
    network = conv_3d(network, 128, 1, activation='relu')
    network = conv_3d(network, 128, 1, activation='relu')
    network = conv_3d(network, 128, 1, activation='relu')
    network = conv_3d(network, 1, [1, 1, 6], activation='relu', padding='valid')
    network = flatten(network)
    network = regression(network, optimizer='momentum', loss='binary_crossentropy')

    model = tflearn.DNN(network)

    return model, network

def get_blocking_model(input_size):

    network_input_size = [None] + list(input_size) + [1] # None 32x 32y 32z 1

    network = input_data(shape=network_input_size)
    network = conv_3d(network, 64, 3, activation='prelu')
    network = max_pool_3d(network, 2, strides=2, padding='same') # 64 16x 16y 16z 1
    network = conv_3d(network, 32, 3, activation='prelu')
    network = max_pool_3d(network, 2, strides=2, padding='same') # 32 8x 8y 8z 1
    network = conv_3d(network, 32, 3, activation='prelu')
    network = conv_3d(network, 1, 1, activation='prelu') # 1 8x 8y 8z 1
    # network = max_pool_3d(network, 2, strides=2, padding='same') # None 4x 4y 4z 1
    # network = conv_3d(network, 32, 3, activation='relu')
    # fully connected
    network = fully_connected(network, 256, activation='prelu')
    network = fully_connected(network, 256, activation='prelu')
    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam', loss='categorical_crossentropy')

    model = tflearn.DNN(network)

    return model, network
