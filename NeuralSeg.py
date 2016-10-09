import sys
import os
import progressbar as pb
import numpy as np
import random
import mtp

sys.path.append('./FlyLIB/')
import neuron

SIZE_TIPS = 27898
# SIZE_TIPS = 20
DIRECTORY_AMS = "/Volumes/toosyou_ext/neuron_data/resampled_111_slow/"


if __name__ == '__main__':

    train_mtp = mtp.MTP('train.mtp')
    print(train_mtp[0].name)
    for coordinates in train_mtp[0].coordinates:
        print(coordinates)

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
