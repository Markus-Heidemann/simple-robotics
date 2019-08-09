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
    # define a controller, which moves the robot and provides the control commands to the localizer
    landmarks_x = [0.0, 100.0, 100.0, 0.0, 75.0]
    landmarks_y = [0.0, 0.0, 100.0, 100.0, 50.0]

    motion_cmd = [MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi),
            MotionCommand(20, 0.25*pi)]

    lm_map = LandmarkMap(landmarks_x, landmarks_y)
    simulator = Simulator(landmark_map=lm_map)

    localizer = Localizer()

    for cmd in motion_cmd:
        simulator.move(cmd.v, cmd.omega)
        localizer.odometry_callback(cmd.v, cmd.omega)

        r, phi = simulator.get_measurements()
        localizer.measurement_callback(r, phi)

    simulator.plot()
    localizer.plot()

    plt.show(block=True)

if __name__ == "__main__":
    main()