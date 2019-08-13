import numpy as np
from VelocityMotionModel import VelocityMotionModel as VMM
from Robot import Pose


class SlamGraph:
    def __init__(self):
        self.mu_0_t = np.array([[0.0, 0.0, 0.0]])
        self.vmm = VMM()

    def initialize(self, motion_commands):
        assert(motion_commands.shape[1] == 2)
        for [v, omega] in motion_commands:
            prev_pose = Pose(self.mu_0_t[-1][0],
                             self.mu_0_t[-1][1], self.mu_0_t[-1][2])
            pose = self.vmm.calculateNewPose(prev_pose, v, omega)
            self.mu_0_t = np.append(self.mu_0_t, np.array(
                [[pose.x, pose.y, pose.theta]]), axis=0)


if __name__ == "__main__":
    pass
