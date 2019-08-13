from matplotlib import pyplot as plt
from math import sqrt
from copy import copy
from collections import namedtuple


class LandmarkMap:
    def __init__(self, x=[], y=[], lm_id=[], color='C5', marker='o'):
        self.plot_color = color
        self.plot_marker = marker
        assert(len(x) == len(y))
        self.x = x
        self.y = y
        if len(lm_id) == 0:
            self.lm_id = list(range(len(x)))
        else:
            assert(len(x) == len(lm_id))
            self.lm_id = lm_id

    def __str__(self):
        ret_str = ""
        for x, y, lm_id in zip(self.x, self.y, self.lm_id):
            ret_str += "ID: {0:>3},\tX: {1:>8},\tY: {2:>8}\n".format(
                lm_id, x, y)
        return ret_str

    def plot(self):
        plt.scatter(self.x, self.y, color=self.plot_color,
                    marker=self.plot_marker)

    def addLandmarks(self, x_list, y_list):
        assert(len(x_list) == len(y_list))
        for x, y in zip(x_list, y_list):
            self.x.append(x)
            self.y.append(y)
            if len(self.lm_id) > 0:
                self.lm_id.append(self.lm_id[-1] + 1)
            else:
                self.lm_id.append(0)

    def getMeasuredLandmarks(self, pose, range):
        x_in_range = []
        y_in_range = []
        for x, y in zip(self.x, self.y):
            r = sqrt((x - pose.x) * (x - pose.x) + (y - pose.y) * (y - pose.y))
            if r <= range:
                x_in_range.append(copy(x))
                y_in_range.append(copy(y))
        return x_in_range, y_in_range


def main():
    pass


if __name__ == "__main__":
    main()
