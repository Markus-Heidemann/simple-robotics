from Robot import Robot, LandmarkMap, ErrorParamsMovement, MotionCommand, ErrorParamMeasurement
from math import pi
from Localizer import Localizer
from matplotlib import pyplot as plt

class Simulator:
    def __init__(self,
                robot=Robot(error_params=ErrorParamsMovement(0.005,0.005,0.0,0.0,0.00005,0.00005)),
                landmark_map=LandmarkMap()):
        self.robot = robot
        self.map = landmark_map

        self.robot.sensor.ep = ErrorParamMeasurement(1.0,0.02)

    def plot(self):
        self.robot.plot()
        self.map.plot()

    # interfaces for the localizer
    def get_measurements(self):
        return self.robot.get_measurements(self.map)

    def move(self, v, omega, t=1):
        self.robot.move(v, omega, t)

def main():
    pass

if __name__ == "__main__":
    main()