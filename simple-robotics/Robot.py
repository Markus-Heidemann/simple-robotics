from math import cos, sin, atan2, pi, degrees, radians, sqrt
import matplotlib.pyplot as plt
from copy import copy
from collections import namedtuple
from numpy.random import normal
from LandmarkMap import LandmarkMap
from VelocityMotionModel import ErrorParamsMovement
from VelocityMotionModel import VelocityMotionModel as VMM

ErrorParamMeasurement = namedtuple(
    'ErrorParamMeasurement', 'sigma_r sigma_phi')
ErrorParamMeasurement.__new__.__defaults__ = (
    0.0,) * len(ErrorParamMeasurement._fields)

MotionCommand = namedtuple('MotionCommand', 'v omega')
MotionCommand.__new__.__defaults__ = (0.0,) * len(MotionCommand._fields)


class Sensor:
    def __init__(self, range=100, error_params=ErrorParamMeasurement()):
        self.range = range
        self.ep = error_params

    def measure(self, pose, world_map):
        x_measurements, y_measurements = world_map.getMeasuredLandmarks(
            pose, self.range)
        r_measurements = []
        phi_measurements = []
        for x, y in zip(x_measurements, y_measurements):
            r = sqrt((x - pose.x) * (x - pose.x) + (y - pose.y) * (y - pose.y))
            phi = atan2(y - pose.y, x - pose.x) - pose.theta
            r_measurements.append(r)
            phi_measurements.append(phi)
        return r_measurements, phi_measurements

    def measure_with_noise(self, pose, world_map):
        r_measurements, phi_measurements = self.measure(pose, world_map)
        r_measurements = [r + normal(0.0, self.ep.sigma_r)
                          for r in r_measurements]
        phi_measurements = [phi + normal(0.0, self.ep.sigma_phi)
                            for phi in phi_measurements]
        return r_measurements, phi_measurements


class Pose:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.x = x
        self.y = y
        self.theta = theta

    def __str__(self):
        return "Pose: X: {0:>8},\tY: {1:>8},\ttheta: {2:>8}".format(self.x, self.y, degrees(self.theta))

    # Round x- and y-values to 3 digits
    # Limit abs(theta) to -2*pi to 2*pi
    def round(self, digits=3):
        self.x = round(self.x, digits)
        self.y = round(self.y, digits)
        while (abs(self.theta) > 2*pi):
            delta = -2*pi if self.theta > 0 else 2*pi
            self.theta += delta
        return self


class Robot:
    def __init__(self,
                 x=0.0,
                 y=0.0,
                 theta=0.0,
                 verbose=False,
                 error_params=ErrorParamsMovement(),
                 color='C0',
                 plot_path=False):
        self.pose = Pose(x, y, theta)
        self.pose_hist = [self.pose]
        self.verbose = verbose
        self.vmm = VMM(error_params)
        self.plot_color = color
        self.plot_path = plot_path
        self.sensor = Sensor()
        self.lm_map = LandmarkMap(color='C2', marker='x')

    def __str__(self):
        return self.pose.__str__()

    def plot(self):
        x = []
        y = []
        for pose in self.pose_hist:
            x.append(pose.x)
            y.append(pose.y)
        plt.scatter(x, y, color=self.plot_color)
        self.lm_map.plot()
        if self.plot_path:
            plt.plot(x, y, color=self.plot_color)

    def move(self, v=0.0, omega=0.0, t=1):
        self.pose = self.vmm.calculateNewPoseWithNoise(self.pose, v, omega, t)
        self.pose_hist.append(copy(self.pose))

        if self.verbose:
            print(self)

    def get_measurements(self, world_map):
        r_meas, phi_meas = self.sensor.measure_with_noise(self.pose, world_map)
        return r_meas, phi_meas

    @classmethod
    def world_t_sensor(self, pose, r_meas, phi_meas):
        x = []
        y = []
        for r, phi in zip(r_meas, phi_meas):
            x.append(pose.x + r * cos(phi + pose.theta))
            y.append(pose.y + r * sin(phi + pose.theta))
        return x, y


def main():
    pass


if __name__ == "__main__":
    main()
