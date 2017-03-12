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
