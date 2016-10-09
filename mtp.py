import numpy as np
import progressbar as pb


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

    def __getitem__(self, index):
        return self.coordinates[index]

class MTP:

    def __init__(self, address_mtp):
        self._tips = list()

        if address_mtp:
            self.read(address_mtp)

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

        print("Reading " + address_mtp + " :")

        pbar = pb.ProgressBar()
        size_tips = int(in_mtp.readline())

        # reading tips
        for i in pbar(range(size_tips)):
            self._tips.append(Points(in_mtp))

        in_mtp.close()
        return True

    def __getitem__(self, index):
        return self._tips[index]
