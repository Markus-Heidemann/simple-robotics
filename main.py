from Robot import MotionCommand
from LandmarkMap import LandmarkMap
from Localizer import Localizer
from Simulator import Simulator
from math import pi
from matplotlib import pyplot as plt
from Robot import Robot, ErrorParamMeasurement, ErrorParamsMovement

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
        x, y = robot_model.world_t_sensor(robot_model.pose, r, phi)
        robot_model.lm_map.addLandmarks(x, y)

    # for _ in range(100):
    #     robot.get_measurements(world_map)

    robot_model.plot()
    robot.plot()

    plt.show(block=True)

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

if __name__ == "__main__":
    main()