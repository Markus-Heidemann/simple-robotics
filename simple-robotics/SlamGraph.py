import numpy as np
from VelocityMotionModel import VelocityMotionModel as VMM
from math import inf, pi, sin, cos


class SlamGraph:
    def __init__(self, init_size_H_b=100):
        self.vmm = VMM()
        self.pose_idx = 1
        self.init_size_H_b = init_size_H_b
        self.H = np.zeros(shape=(SlamGraph.init_size_H_b * 3, SlamGraph.init_size_H_b * 3))
        self.b = np.zeros(shape=(SlamGraph.init_size_H_b * 3))


    def add_pose_landmark_edge(self, x, l, i, j, z, Omega_ij=np.identity(3)):
        e, A, B = self.linearize_pose_landmark_constraint(x, l, z)

        b_i = e.transpose().dot(Omega_ij).dot(A)
        b_j = e.transpose().dot(Omega_ij).dot(B)
        self.b[i*3 : i*3 + 2] = b_i
        self.b[j*3 : j*3 + 2] = b_j

        Hii = A.transpose().dot(Omega_ij).dot(A)
        Hij = A.transpose().dot(Omega_ij).dot(B)
        Hji = B.transpose().dot(Omega_ij).dot(A)
        Hjj = B.transpose().dot(Omega_ij).dot(B)

        self.H[i*3 : i*3 + 2, i*3 : i*3 + 2] = Hii
        self.H[i*3 : i*3 + 2, j*3 : j*3 + 2] = Hij
        self.H[j*3 : j*3 + 2, i*3 : i*3 + 2] = Hji
        self.H[j*3 : j*3 + 2, j*3 : j*3 + 2] = Hjj

        self.pose_idx += 1


    def add_pose_pose_edge(self, x1, x2, z=np.array([0, 0, 0]), Omega_ij=np.identity(3)):
        i = self.pose_idx - 1
        j = self.pose_idx

        self.add_pose_landmark_edge(x1, x2, i, j, z, Omega_ij)


    def linearize_pose_landmark_constraint(self, x, l, z):
        A = np.zeros(shape=(3, 3))
        A[0][0] = -cos(x[2])
        A[0][1] = -sin(x[2])
        A[0][2] = (-1) * (l[0] - x[0]) * sin(x[2]) + (l[1] - x[1]) * cos(x[2])
        A[1][0] = sin(x[2])
        A[1][1] = -cos(x[2])
        A[1][2] = -(l[0] - x[0]) * cos(x[2]) - (l[1] - x[1]) * sin(x[2])

        B = np.zeros(shape=(3, 3))
        B[0][0] = cos(x[2])
        B[0][1] = sin(x[2])
        B[1][0] = -sin(x[2])
        B[1][1] = cos(x[2])

        e_0 = (l[0] - x[0]) * cos(x[2]) + (l[1] - x[1]) * sin(x[2]) - z[0]
        e_1 = -(l[0] - x[0]) * sin(x[2]) + (l[1] - x[1]) * cos(x[2]) - z[1]
        e = np.array([e_0, e_1, 0])

        return e, A, B


    def linearize_pose_pose_constraint(self, x1, x2, z):
        A = np.ndarray(shape=(3, 3))
        A[0][0] = -cos(x1[2]) * cos(z[2]) + sin(x1[2]) * sin(z[2])
        A[0][1] = -sin(x1[2]) * cos(z[2]) - cos(x1[2]) * sin(z[2])
        A[0][2] = (-sin(x1[2]) * cos(z[2]) - cos(x1[2]) * sin(z[2])) * (x2[0] - x1[0]) + \
                (cos(x1[2]) * cos(z[2]) - sin(x1[2]) * sin(z[2])) * (x2[1] - x1[1])
        A[1][0] = cos(x1[2]) * sin(z[2]) + sin(x1[2]) * cos(z[2])
        A[1][1] = sin(x1[2]) * sin(z[2]) - cos(x1[2]) * cos(z[2])
        A[1][2] = (sin(z[2]) * sin(x1[2]) - cos(z[2]) * cos(x1[2])) * (x2[0] - x1[0]) + \
                (-sin(z[2]) * cos(x1[2]) - cos(z[2]) * sin(x1[2])) * (x2[1] - x1[1])
        A[2][0] = 0
        A[2][1] = 0
        A[2][2] = -1

        B = np.ndarray(shape=(3, 3))
        B[0][0] = cos(x1[2]) * cos(z[2]) - sin(x1[2]) * sin(z[2])
        B[0][1] = sin(x1[2]) * cos(z[2]) + cos(x1[2]) * sin(z[2])
        B[0][2] = 0
        B[1][0] = -cos(x1[2]) * sin(z[2]) - sin(x1[2]) * cos(z[2])
        B[1][1] = -sin(x1[2]) * sin(z[2]) + cos(x1[2]) * cos(z[2])
        B[1][2] = 0
        B[2][0] = 0
        B[2][1] = 0
        B[2][2] = 1

        R_i_j = np.ndarray(shape=(2, 2))
        R_i = np.ndarray(shape=(2, 2))
        R_i_j[0][0] = cos(z[2])
        R_i_j[0][1] = -sin(z[2])
        R_i_j[1][0] = sin(z[2])
        R_i_j[1][1] = cos(z[2])
        R_i[0][0] = cos(x1[2])
        R_i[0][1] = -sin(x1[2])
        R_i[1][0] = sin(x1[2])
        R_i[1][1] = cos(x1[2])

        t_i_j = np.ndarray(shape=(2,1))
        t_i = np.ndarray(shape=(2,1))
        t_j = np.ndarray(shape=(2,1))
        t_i_j[0] = z[0]
        t_i_j[1] = z[1]
        t_i[0] = x1[0]
        t_i[1] = x1[1]
        t_j[0] = x2[0]
        t_j[1] = x2[1]

        e_i_j_01 = R_i_j.transpose().dot(R_i.transpose().dot(t_j - t_i) - t_i_j)
        e_i_j_2 = x2[2] - x1[2] - z[2]
        e = np.array([e_i_j_01[0][0], e_i_j_01[1][0], e_i_j_2])

        return e, A, B


if __name__ == "__main__":
    motion_cmds = np.array([[1, 0],
                            [0, 0.5*pi],
                            [1, 0],
                            [0, -0.5*pi],
                            [1, 0]])
    slam_graph = SlamGraph()
