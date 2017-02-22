import scipy
import numpy as np
import progressbar as pb
import os
import copy
import progressbar as pb
from sklearn.utils import shuffle
from random import random
import tqdm
from tqdm import tqdm, trange

import sys
sys.path.append('./FlyLIB/')
import neuron

EMPTY_SLICE_RATIO = 0.1 # 10% of slices data with empty output

DIRECTORY_AMS = "/home/toosyou/ext/neuron_data/resampled_111_slow/" # must contains the ending slash

def get_am_filename(neuron_name):
    index_underline = neuron_name.rindex('_')
    rtn = neuron_name[0:index_underline+1] + 'resampled_1_1_1_ascii.am'
    return rtn;

class Points:

    def __init__(self, in_mtp):
        self.name = str()
        self.coordinates = np.array([])

        if in_mtp:
            self.read(in_mtp)

    def read(self, in_mtp):

        self.name = in_mtp.readline()[:-1]

        size_coordinates = int(in_mtp.readline())
        self.coordinates = np.ndarray(shape=(size_coordinates, 3), dtype=float)

        # read tips coordinate
        for i in range(size_coordinates):
            line = in_mtp.readline()
            self.coordinates[i] = list(map(float, line.split()))

    def am_name(self):
        return get_am_filename(self.name)

    def am_exists(self):
        try:
            in_am = open(DIRECTORY_AMS+self.am_name(), 'r')
        except IOError: # cannot be opened
            return False
        in_am.close()
        return True

    def __getitem__(self, index):
        return self.coordinates[index]

class MTP:

    def __init__(self, address_mtp, clean_up=True):
        self._tips = list()

        if address_mtp:
            self.read(address_mtp)
            if clean_up:
                self.clean_not_exist()

    def write(self, address_mtp):
        out_mtp = open(address_mtp, 'w')
        out_mtp.write(str(len(self._tips)) + '\n') # total number of neurons

        print("Writing " + address_mtp + " :")
        pbar = pb.ProgressBar()
        for tips in pbar(self._tips):
            out_mtp.write(tips.neuron_am_name + '\n')
            out_mtp.write(str(len(tips.coordinates)) + '\n')
            for coordinates in tips.coordinates:
                out_mtp.write(str(coordinates[0]) + ' ')
                out_mtp.write(str(coordinates[1]) + ' ')
                out_mtp.write(str(coordinates[2]) + '\n')

        out_mtp.close()

    def read(self, address_mtp):
        try:
            in_mtp = open(address_mtp, 'r')
        except IOError:
            return False

        size_tips = int(in_mtp.readline())

        # reading tips
        for i in trange(size_tips, desc='Reading '+address_mtp):
            self._tips.append(Points(in_mtp))

        in_mtp.close()
        return True

    def size(self):
        return len(self._tips)

    def clean_not_exist(self):
        number_removed = 0
        with tqdm(total=self.size(), desc='Clean up') as progressbar:
            for i, points in enumerate(list(self._tips)):
                if points.am_exists() == False:
                    self._tips.pop( i - number_removed )
                    number_removed += 1
                progressbar.update()
        return

    def read_neuron(self, index):
        points = self._tips[index]
        am_name = get_am_filename(points.name)
        return neuron.NeuronRaw(am_name)

    def get_target(self, neuron, index):
        target = neuron.copy()
        target.clean_intensity()
        target.read_from_points(self._tips[index].coordinates)
        return target

    def get_sliced_data(self, index_start, size, size_input, size_output):
        X = list()
        Y = list()

        # change directory to ams
        original_directory = os.getcwd()
        os.chdir(DIRECTORY_AMS)

        # read size sliced data
        progressbar = tqdm(total=size, desc='Slice '+str(index_start))
        number_succeed_read = 0
        index_so_far = index_start
        while number_succeed_read < size:
            # read neural raw data
            raw = self.read_neuron(index_so_far)
            if raw.valid == False: # cannot read from am_name
                # print("*****ERROR READING: ", am_name, " *****", file=sys.stderr)
                continue
            else:
                number_succeed_read += 1

            # get target for Y
            target = self.get_target(raw, index_so_far)

            # resize raw and target data xy to fit size_input
            raw.resize([size_input[0], size_input[1], -1], order=1) # bilinear
            target.resize([size_output[0], size_output[1], -1], order=0) # nestest

            # slice raw by z-axis
            for z in range(raw.size[2]):
                slices = raw.block(start_point=(0, 0, z-10), size=size_input)

                # pass empty target
                if not np.count_nonzero(target.intensity[:,:,z]): # target's empty, continue
                    # 10% empty data
                    if random() < EMPTY_SLICE_RATIO:
                        X.append(slices.copy().intensity)
                        Y.append(copy.deepcopy(target.intensity[:,:,z]).flatten())
                    continue

                # apply rotate and mirron to both x and y
                copy_slices = slices.copy().intensity
                copy_target = copy.deepcopy(target.intensity[:, :, z])
                for rotate in range(4):
                    for mirron in range(2):
                        # rotate 'rotate' times
                        this_x = np.rot90(copy_slices, k=rotate, axes=[0, 1])
                        this_y = np.rot90(copy_target, k=rotate, axes=[0, 1])
                        # mirron 'mirron' times
                        if mirron != 0:
                            this_x = this_x[:,::-1,:]
                            this_y = this_y[:,::-1]
                        X.append(this_x)
                        Y.append(this_y.flatten())

            # increase index
            index_so_far = (index_so_far+1) % self.size()
            # update progressbar
            progressbar.update()


        # shuffle X, Y and make X and Y np-arrays
        X, Y = shuffle(X, Y)
        X = np.array(X)
        Y = np.array(Y)

        # change directory back to original one
        os.chdir(original_directory)

        return X, Y, index_so_far

    def get_raw_data(self, index_start, size, size_input, size_output):
        X = list()
        Y = list()

        # change directory to ams
        original_directory = os.getcwd()
        os.chdir(DIRECTORY_AMS)

        # get 'size' neurons
        print('Geting data:')
        progressbar = pb.ProgressBar(max_value=size)
        number_succeed_read = 0
        passed_offset = 0
        while number_succeed_read < size:
            # calculate index
            this_index = ( index_start + number_succeed_read + passed_offset) % self.size()
            points = self._tips[this_index]
            am_name = get_am_filename(points.name)
            # read neural raw data
            raw = neuron.NeuronRaw(am_name)
            if raw.valid == False: # cannot read from am_name
                print("*****ERROR READING: ", am_name, " *****", file=sys.stderr)
                passed_offset += 1
                continue
            else:
                number_succeed_read += 1
                progressbar.update(number_succeed_read)

            # get input data X, resize to size_input and normalize
            this_x = copy.deepcopy(raw.resize(size_input, copy=True).intensity.reshape(size_input.append(1))) # 200 200 200 1
            X.append(this_x)
            # get output data Y, resize to size_output and flatten it
            raw.clean_intensity()
            raw.read_from_points(points.coordinates)
            raw.resize(size_output)
            this_y = copy.deepcopy( raw.intensity.flatten() )
            Y.append(this_y)

        # make X and Y np-array
        X = np.array(X)
        Y = np.array(Y)

        # change directory back to original one
        os.chdir(original_directory)
        return X, Y, (index_start + size + passed_offset) % self.size()

    def __getitem__(self, index):
        return self._tips[index]
