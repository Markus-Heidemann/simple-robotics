from math import cos, sin, atan2, pi, degrees, radians, sqrt
import matplotlib.pyplot as plt
from copy import copy
from collections import namedtuple
from numpy.random import normal

ErrorParamsMovement = namedtuple('ErrorParamsMovement', 'a0 a1 a2 a3 a4 a5')
ErrorParamMeasurement = namedtuple('ErrorParamMeasurement', 'sigma_r sigma_phi')
MotionCommand = namedtuple('MotionCommand', 'v omega')
Landmark = namedtuple('Landmark', 'id x y')

class Sensor:
    def __init__(self, range=100, error_params=ErrorParamMeasurement(1.0,0.02)):
        self.range = range
        self.ep = error_params

    def measure(self, pose, world_map):
        x_measurements, y_measurements = world_map.getMeasuredLandmarks(pose, self.range)
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
        r_measurements = [r + normal(0.0, self.ep.sigma_r) for r in r_measurements]
        phi_measurements = [phi + normal(0.0, self.ep.sigma_phi) for phi in phi_measurements]
        return r_measurements, phi_measurements


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
        for x,y,lm_id in zip(self.x, self.y, self.lm_id):
            ret_str += "ID: {0:>3},\tX: {1:>8},\tY: {2:>8}\n".format(lm_id, x, y)
        return ret_str

    def plot(self):
        plt.scatter(self.x, self.y, color=self.plot_color, marker=self.plot_marker)

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
                error_params=ErrorParamsMovement(0.0,0.0,0.0,0.0,.0,0.0),
                color='C0',
                plot_path=False):
        self.pose = Pose(x, y, theta)
        self.pose_hist = [self.pose]
        self.verbose = verbose
        self.ep = error_params
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

    # Positive 'v' --> forward movement
    # Positive 'omega' --> counter-clockwise rotation
    def calculateNewPose(self, pose, v=0.0, omega=0.0, t=1):
        if omega == 0.0:
            pose.y += v * t * sin(pose.theta)
            pose.x += v * t * cos(pose.theta)
        else:
            n = v / omega
            pose.x += n * (-sin(pose.theta) + sin(pose.theta + omega * t))
            pose.y += n * (cos(pose.theta) - cos(pose.theta + omega * t))
            pose.theta += omega * t
        return pose

    def move(self, v=0.0, omega=0.0, t=1):
        sigma_v = self.ep.a0 * v * v + self.ep.a1 * omega * omega
        sigma_omega = self.ep.a2 * v * v + self.ep.a3 * omega * omega
        sigma_theta = self.ep.a4 * v * v + self.ep.a5 * omega * omega

        v += normal(0.0, sigma_v)
        omega += normal(0.0, sigma_omega)

        pose = self.calculateNewPose(self.pose, v, omega, t)
        pose.theta += normal(0.0, sigma_theta) * t

        self.pose = pose.round()
        self.pose_hist.append(copy(self.pose))

        if self.verbose: print(self)

    def get_measurements(self, world_map):
        r_meas, phi_meas = self.sensor.measure_with_noise(self.pose, world_map)
        return r_meas, phi_meas

    def world_t_sensor(self, r_meas, phi_meas):
        x = []
        y = []
        for r, phi in zip(r_meas, phi_meas):
            x.append(self.pose.x + r * cos(phi + self.pose.theta))
            y.append(self.pose.y + r * sin(phi + self.pose.theta))
        return x, y

def plot_motion_model_distribution():
    motion_cmd = [MotionCommand(1, 0.25*pi),
                MotionCommand(1, 0.25*pi),
                MotionCommand(1, 0.25*pi),
                MotionCommand(1, 0.25*pi),
                MotionCommand(1, 0.25*pi),
                MotionCommand(1, 0.25*pi),
                MotionCommand(1, 0.25*pi)]

    ep = ErrorParamsMovement(0.0,0.0,0.0,0.0,0.05,0.05)

    # Plot paths of 100 noisy robots
    for _ in range(100):
        robot_noise = Robot(0, 0, 0, False, error_params=ep, color='C4')
        for command in motion_cmd:
            robot_noise.move(command.v, command.omega)
        robot_noise.plot()

    # Plot motion of ideal motion model
    robot = Robot(0, 0, 0, False, color='C0')
    for command in motion_cmd:
        robot.move(command.v, command.omega)
    robot.plot()

    plt.show(block=True)

def test_map():
    landmarks_x = [0.0, 100.0, 100.0, 0.0, 75.0]
    landmarks_y = [0.0, 0.0, 100.0, 100.0, 50.0]
    landmarks_id = [0, 1, 2, 3, 4]

    world_map = LandmarkMap(landmarks_x, landmarks_y, landmarks_id)
    world_map.plot()

    motion_cmd = [MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi)]

    # ep = ErrorParamsMovement(0.0,0.0,0.0,0.0,0.0,0.0)
    ep = ErrorParamsMovement(0.005,0.005,0.0,0.0,0.00005,0.00005)
    robot = Robot(0, 0, 0, False, color='C0', error_params=ep)
    robot_model = Robot(0, 0, 0, False, color='C6')
    for cmd in motion_cmd:
        robot.move(cmd.v, cmd.omega)
        robot_model.move(cmd.v, cmd.omega)
        r, phi = robot.get_measurements(world_map)
        x, y = robot_model.world_t_sensor(r, phi)
        robot_model.lm_map.addLandmarks(x, y)

    # for _ in range(100):
    #     robot.get_measurements(world_map)

    robot_model.plot()
    robot.plot()

    plt.show(block=True)

def main():
    test_map()
    # plot_motion_model_distribution()

if __name__ == "__main__":
    main()