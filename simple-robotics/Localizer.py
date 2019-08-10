from Robot import LandmarkMap, Robot, Pose

class Localizer:
    def __init__(self):
        self.map = LandmarkMap(color='C7', marker='x')
        self.robot_model = Robot(color='C7')

    def measurement_callback(self, r, phi):
        x, y = Robot.world_t_sensor(self.robot_model.pose, r, phi)
        self.map.addLandmarks(x, y)

    def odometry_callback(self, v, omega, t=1):
        self.robot_model.move(v, omega, t)

    def plot(self):
        self.robot_model.plot()
        self.map.plot()

def main():
    pass

if __name__ == "__main__":
    main()