import sys
import os
import progressbar as pb
import numpy as np

sys.path.append('./FlyLIB/')
import neuron

SIZE_TIPS = 27898
# SIZE_TIPS = 20
DIRECTORY_AMS = "/Volumes/toosyou_ext/neuron_data/resampled_111_slow/"


class Points:

    def __init__(self, address_pts):
        self.neuron_am_name = str()
        self.coordinates = np.array([])

        if address_pts:
            self.read_tips(address_pts)

    def read_tips(self, address_pts):
        in_pts = open(address_pts, 'r')
        line = "init_str"

        self.neuron_am_name = in_pts.readline()[:-1]
        coordinates = list()

        # read tips coordinate
        while not(not line):
            line = in_pts.readline()
            if len(line) != 0:
                coordinates.append(list(map(float, line.split())))

        self.coordinates = np.array(coordinates)
        in_pts.close()


def read_tips(dir_tips, size_tips):
    tips = list()

    # change directory
    original_directory = os.getcwd()
    os.chdir(dir_tips)

    # read .tips one by one
    print("Reading tips:")
    pbar = pb.ProgressBar()
    for i in pbar(range(size_tips)):
        tips_file_name = str(i) + str('.tips')
        tips.append(Points(tips_file_name))

    # change directory back
    os.chdir(original_directory)

    return tips

if __name__ == '__main__':

    tips = read_tips('tips_branches', SIZE_TIPS)
    neurons = list()

    # read am files
    original_directory = os.getcwd()
    os.chdir(DIRECTORY_AMS)
    pbar = pb.ProgressBar()
    for t in pbar(tips[0:20]):
        neuron_name = t.neuron_am_name[:-7] + 'resampled_1_1_1_ascii.am'
        tmp_neuron = neuron.NeuronRaw( neuron_name )
        if tmp_neuron.valid == True:
            neurons.append( tmp_neuron )

    os.chdir(original_directory)
