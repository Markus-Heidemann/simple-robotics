from math import pi, sin, cos
from collections import namedtuple
from numpy.random import normal

ErrorParamsMovement = namedtuple('ErrorParamsMovement', 'a0 a1 a2 a3 a4 a5')
ErrorParamsMovement.__new__.__defaults__ = (0.0,) * len(ErrorParamsMovement._fields)

class VelocityMotionModel:
    def __init__(self, ep=ErrorParamsMovement()):
        self.ep = ep

    # Calculates exact new pose given a motion command
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

    def calculateNewPoseWithNoise(self, pose, v=0.0, omega=0.0, t=1):
        sigma_v = self.ep.a0 * v * v + self.ep.a1 * omega * omega
        sigma_omega = self.ep.a2 * v * v + self.ep.a3 * omega * omega
        sigma_theta = self.ep.a4 * v * v + self.ep.a5 * omega * omega

        v += normal(0.0, sigma_v)
        omega += normal(0.0, sigma_omega)

        pose = self.calculateNewPose(pose, v, omega, t)
        pose.theta += normal(0.0, sigma_theta) * t

        pose = pose.round()
        return pose


if __name__ == "__main__":
    pass