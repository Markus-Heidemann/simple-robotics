import numpy as np
from VelocityMotionModel import VelocityMotionModel as VMM
from Robot import Pose
from scipy.sparse import bsr_matrix
from math import inf, pi, sin, cos


class SlamGraph:
    def __init__(self):
        self.mu_0_t = np.array([[0.0, 0.0, 0.0]])
        self.vmm = VMM()

    def initialize(self, motion_commands):
        assert(motion_commands.shape[1] == 3)
        for v, omega, t in motion_commands:
            prev_pose = Pose(self.mu_0_t[-1][0],
                             self.mu_0_t[-1][1], self.mu_0_t[-1][2])
            pose = self.vmm.calculateNewPose(prev_pose, v, omega, t)
            self.mu_0_t = np.append(self.mu_0_t, np.array(
                [[pose.x, pose.y, pose.theta]]), axis=0)

    def linearize(self, motion_commands):
        assert(motion_commands.shape[1] == 3)
        num_cmds = motion_commands.shape[0]

        info_mat = np.ndarray(shape=(num_cmds * 3, num_cmds * 3))
        xi = np.array(shape=(num_cmds * 3))

        info_mat[0][0] = inf
        info_mat[1][1] = inf
        info_mat[2][2] = inf

        for i, cmd in enumerate(motion_commands):
            v, omega, t = cmd[0]
            mu_x, mu_y, mu_theta = self.mu_0_t[i-1]

            x_hat_x = mu_x + ((v/omega) * (-sin(mu_theta) + sin(mu_theta + omega * t)))
            x_hat_y = mu_y + ((v/omega) * (cos(mu_theta) - cos(mu_theta + omega * t)))
            x_hat_theta = mu_theta + (omega * t)

            x_hat = np.array([x_hat_x, x_hat_y, x_hat_theta])

            G_t_0_2 = (v/omega) * (-cos(mu_theta) + cos(mu_theta + omega * t))
            G_t_1_2 = (v/omega) * (-sin(mu_theta) + sin(mu_theta + omega * t))

            G_t = np.ndarray(shape=(3,3))
            G_t[0][0] = 1
            G_t[1][1] = 1
            G_t[2][2] = 1
            G_t[0][2] = G_t_0_2
            G_t[1][2] = G_t_1_2

    def linearize_pose_landmark_constraint(self, x, l, z):
        A = np.ndarray(shape=(2, 3))
        A[0][0] = -cos(x[2])
        A[0][1] = -sin(x[2])
        A[0][2] = (-1) * (l[0] - x[0]) * sin(x[2]) + (l[1] - x[1]) * cos(x[2])
        A[1][0] = sin(x[2])
        A[1][1] = -cos(x[2])
        A[1][2] = -(l[0] - x[0]) * cos(x[2]) - (l[1] - x[1]) * sin(x[2])

        B = np.ndarray(shape=(2, 2))
        B[0][0] = cos(x[2])
        B[0][1] = sin(x[2])
        B[1][0] = -sin(x[2])
        B[1][1] = cos(x[2])

        e_0 = (l[0] - x[0]) * cos(x[2]) + (l[1] - x[1]) * sin(x[2]) - z[0]
        e_1 = -(l[0] - x[0]) * sin(x[2]) + (l[1] - x[1]) * cos(x[2]) - z[1]
        e = np.array([e_0, e_1])

        return e, A, B

if __name__ == "__main__":
    motion_cmds = np.array([[1, 0],
                            [0, 0.5*pi],
                            [1, 0],
                            [0, -0.5*pi],
                            [1, 0]])
    slam_graph = SlamGraph()
    slam_graph.linearize(motion_cmds)
