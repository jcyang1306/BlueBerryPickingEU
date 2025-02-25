### General robot kinematics utils
### code credit: CURI team
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from eu_arm_const import *

#####robot class #####
class robot:
    # basic parameters
    JOINT_SIZE = 0
    A = []
    ALPHA = []
    D = []
    THETA = []
    JOINT_TYPE = []
    # other parameters to be added by users

    def __init__(self, joint_size=JNT_NUM, joint_type=kJOINT_TYPE, a=kDH_A, alpha=kDH_ALPHA, d=kDH_D, theta=kDH_THETA):
        self.JOINT_SIZE = joint_size
        self.A = a
        self.ALPHA = alpha
        self.D = d
        self.THETA = theta
        if joint_type == []:
            self.JOINT_TYPE = numpy.zeros((joint_size))
        else:
            self.JOINT_TYPE = numpy.array(joint_type)
        return

    def A1(self, theta, d):
        ans = numpy.array([[+numpy.cos(theta), -numpy.sin(theta), 0, 0],
                           [+numpy.sin(theta), +numpy.cos(theta), 0, 0],
                           [                0,                 0, 1, d],
                           [                0,                 0, 0, 1]])
        return ans

    def A2(self, alpha, a):
        ans = numpy.array([[1,                 0,                 0, a],
                           [0, +numpy.cos(alpha), -numpy.sin(alpha), 0],
                           [0, +numpy.sin(alpha), +numpy.cos(alpha), 0],
                           [0,                 0,                 0, 1]])
        return ans

    def sk(self, w):
        return numpy.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])

    # standard DH method (Siciliano's book)
    #
    # i-1         i         
    #  +----------+  Oi-1
    #             |         i+1
    #             +----------+  Qi 
    #                       
    def SDH(self, a, alpha, d, theta):
        return numpy.dot(self.A1(theta, d), self.A2(alpha, a))

    def SFK(self, theta):
        T = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for k in range(0, self.JOINT_SIZE):
            if self.JOINT_TYPE[k] == 0:
                T = numpy.dot(T, self.SDH(self.A[k], self.ALPHA[k], self.D[k], self.THETA[k]+theta[k]))
            else:
                T = numpy.dot(T, self.SDH(self.A[k], self.ALPHA[k], self.D[k]+theta[k], self.THETA[k]))
        return T

    def SDK(self, theta):
        T = numpy.zeros((self.JOINT_SIZE+1, 4, 4))
        J = numpy.zeros((6, self.JOINT_SIZE))
        T[0] = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for k in range(0, self.JOINT_SIZE):
            T[k+1] = numpy.dot(T[k], self.SDH(self.A[k], self.ALPHA[k], self.D[k], theta[k]))
        for k in range(0, self.JOINT_SIZE):
            if self.JOINT_TYPE[k] == 0:
                J[0:3, k] = numpy.cross(T[k][0:3, 2], T[self.JOINT_SIZE][0:3, 3] - T[k][0:3, 3])
                J[3:6, k] = T[k][0:3, 2]
            else:
                J[0:3, k] = T[k][0:3, 2]
        return J

    def SIK(self, Rt, Pt, q, iterate_times = 100):
        q_ans = q.copy()
        Tc = self.SFK(q)
        error = Pt - Tc[0:3, 3]
        count = 0
        while numpy.linalg.norm(error) > 2e-5 and count < iterate_times:
            J = self.SDK(q)
            if numpy.linalg.matrix_rank(J) < 6:
                print('singularity')
                return q_ans
            dv = error
            dw = 0.5*(numpy.cross(Tc[0:3, 0], Rt[0:3, 0]) + numpy.cross(Tc[0:3, 1], Rt[0:3, 1]) + numpy.cross(Tc[0:3, 2], Rt[0:3, 2]))

            dx = numpy.array([dv[0], dv[1], dv[2], dw[0], dw[1], dw[2]])*0.5
            dq = numpy.dot(numpy.linalg.pinv(J), dx)
            q = q + dq.flatten()

            Tc = self.SFK(q)
            error = Pt - Tc[0:3, 3]
            count = count + 1

        if count >= iterate_times:
            print('iterates more than ' + str(iterate_times) + ' times')
            return q_ans
        return q

    # modify DH method (Creig's book)
    #
    # i-1         i         
    #  +----------+  Oi
    #             |         i+1
    #             +----------+  Qi+1  
    #                       
    def MDH(self, a, alpha, d, theta):
        return numpy.dot(self.A2(alpha, a), self.A1(theta, d))

    def MFK(self, theta):
        T = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for k in range(0, self.JOINT_SIZE):
            if self.JOINT_TYPE[k] == 0:
                T = numpy.dot(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k], self.THETA[k]+theta[k]))
            else:
                T = numpy.dot(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k]+theta[k], self.THETA[k]))
            # print(f'T[{k}] = \n{T}')
        return T

    def MDK(self, theta):
        Te = self.MFK(theta)
        T = numpy.zeros((4, 4))
        J = numpy.zeros((6, self.JOINT_SIZE))
        T = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        for k in range(0, self.JOINT_SIZE):
            if self.JOINT_TYPE[k] == 0:
                T = numpy.dot(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k], self.THETA[k]+theta[k]))
                J[0:3, k] = numpy.cross(T[0:3, 2], Te[0:3, 3] - T[0:3, 3])
                J[3:6, k] = T[0:3, 2]
            else:
                T = numpy.dot(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k]+theta[k], self.THETA[k]))
                J[0:3, k] = T[0:3, 2]
        return J

    def MIK(self, Rt, Pt, q, iterate_times=100):
        q_ans = q.copy()
        Tc = self.MFK(q)
        error = Pt - Tc[0:3, 3]
        count = 0
        while numpy.linalg.norm(error) > 1e-4 and count < iterate_times:
            J = self.MDK(q)
            if numpy.linalg.matrix_rank(J) < 6:
                print('singularity')
                return q_ans
            dv = error
            dw = 0.5*(numpy.cross(Tc[0:3, 0], Rt[0:3, 0]) + numpy.cross(Tc[0:3, 1], Rt[0:3, 1]) + numpy.cross(Tc[0:3, 2], Rt[0:3, 2]))

            dx = numpy.array([dv[0], dv[1], dv[2], dw[0], dw[1], dw[2]])*0.5
            dq = numpy.dot(numpy.linalg.pinv(J), dx)
            q = q + dq.flatten()

            Tc = self.MFK(q)
            error = Pt - Tc[0:3, 3]
            count = count + 1

        if count >= iterate_times:
            print('iterates more than ' + str(iterate_times) + ' times')
            return q_ans
        return q

    from utils.benchmark import benchmark
    @benchmark
    def MIK_from_T(self, T_desired, q, iterate_times=100):
        Rt = T_desired[0:3, 0:3]
        Pt = T_desired[0:3, 3]
        return self.MIK(Rt, Pt, q, iterate_times)

    def mdraw(self, q, fig, show_frame=False):
        scale = 0.01
        plt.cla()
        ax = Axes3D(fig)
        ax.bar3d(-5*scale, -5*scale, -3*scale, 10*scale, 10*scale, 3*scale, color='gray')
        T = numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        P = T[0:3, 3]
        R = T[0:3, 0:3]
        ax.quiver(-0.3, 0.3, 0, R[0, 0]*scale*5, R[1, 0]*scale*5, R[2, 0]*scale*5, length=0.1, normalize=True, color='r')
        ax.quiver(-0.3, 0.3, 0, R[0, 1]*scale*5, R[1, 1]*scale*5, R[2, 1]*scale*5, length=0.1, normalize=True, color='g')
        ax.quiver(-0.3, 0.3, 0, R[0, 2]*scale*5, R[1, 2]*scale*5, R[2, 2]*scale*5, length=0.1, normalize=True, color='b')
        ax.text(-0.3 + R[0, 0]*scale*15, 0.3 + R[1, 0]*scale*15, R[2, 0]*scale*15, 'x')
        ax.text(-0.3 + R[0, 1]*scale*15, 0.3 + R[1, 1]*scale*15, R[2, 1]*scale*15, 'y')
        ax.text(-0.3 + R[0, 2]*scale*10, 0.3 + R[1, 2]*scale*10, R[2, 2]*scale*10, 'z')
        for k in range(0, self.JOINT_SIZE):
            T = numpy.dot(T, self.MDH(self.A[k], self.ALPHA[k], self.D[k], q[k]))
            Q = P
            P = T[0:3, 3]
            R = T[0:3, 0:3]
            # draw line
            ax.plot([Q[0], P[0]], [Q[1], P[1]], [Q[2], P[2]], linewidth=5, color='orange')
            
            # draw cylinder
            n = 20
            u = numpy.linspace(0, 2*numpy.pi, n)
            x = numpy.array([numpy.cos(u)*scale*2, numpy.cos(u)*scale*2])
            y = numpy.array([numpy.sin(u)*scale*2, numpy.sin(u)*scale*2])
            z = numpy.array([[-scale*4]*n, [+scale*4]*n])

            for j in range(0, n):
                xx0 = R[0, 0]*x[0][j] + R[0, 1]*y[0][j] + R[0, 2]*z[0][j] + P[0]
                yy0 = R[1, 0]*x[0][j] + R[1, 1]*y[0][j] + R[1, 2]*z[0][j] + P[1]
                zz0 = R[2, 0]*x[0][j] + R[2, 1]*y[0][j] + R[2, 2]*z[0][j] + P[2]

                xx1 = R[0, 0]*x[1][j] + R[0, 1]*y[1][j] + R[0, 2]*z[1][j] + P[0]
                yy1 = R[1, 0]*x[1][j] + R[1, 1]*y[1][j] + R[1, 2]*z[1][j] + P[1]
                zz1 = R[2, 0]*x[1][j] + R[2, 1]*y[1][j] + R[2, 2]*z[1][j] + P[2]

                x[0][j] = xx0
                y[0][j] = yy0
                z[0][j] = zz0

                x[1][j] = xx1
                y[1][j] = yy1
                z[1][j] = zz1
            ax.plot_surface(x, y, z, color='lightblue')
            
            # draw coordinate
            if show_frame == True or k == self.JOINT_SIZE-1:
                ax.quiver(P[0], P[1], P[2], R[0, 0]*scale*5, R[1, 0]*scale*5, R[2, 0]*scale*5, length=0.1, normalize=True, color='r')
                ax.quiver(P[0], P[1], P[2], R[0, 1]*scale*5, R[1, 1]*scale*5, R[2, 1]*scale*5, length=0.1, normalize=True, color='g')
                ax.quiver(P[0], P[1], P[2], R[0, 2]*scale*5, R[1, 2]*scale*5, R[2, 2]*scale*5, length=0.1, normalize=True, color='b')

        #ax.set_xlabel('X (m)')
        ax.set_xlim(-0.5, 0.5)
        #ax.set_ylabel('Y (m)')
        ax.set_ylim(-0.5, 0.5)
        #ax.set_zlabel('Z (m)')
        ax.set_zlim(-0.3, 0.7)
        #ax.view_init(azim=135)
        plt.axis('off')
        return ax

def test_eyou_sdh():
    # 定义DH参数
    joint_size = 6
    joint_type = [0, 0, 0, 0, 0, 0]  # 全部为转动关节
    a = numpy.array([0, 183.765, 164.6548, -0.4715, -1.7394, 0]) * 1e-3
    alpha = numpy.array([90, 179.3582, 179.6965, -90.2676, 90.0063, 0]) * numpy.pi / 180.
    d = numpy.array([109, 0.7599, -0.1203, 76.2699, 73.7185, 69.6]) * 1e-3
    theta = numpy.array([-0.2988, 88.8208, 0.0709, -90.3032, -0.0696, 0.0002]) * numpy.pi / 180.
    # 初始化机器人
    my_robot = robot(joint_size, joint_type, a, alpha, d, theta)

    # 定义关节角度
    q_initial = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # 计算末端位姿
    end_effector = my_robot.SFK(q_initial)
    print("s末端位姿:\n", end_effector)

def test_eyou_mdh(): 
    # 定义DH参数
    joint_size = 6
    joint_type = [0, 0, 0, 0, 0, 0]  # 全部为转动关节
    a = numpy.array([0, 0, 183.765, 164.6548, -0.4715, -1.7394]) * 1e-3
    alpha = numpy.array([0, 90, 179.3582, 179.6965, -90.2676, 90.0063]) * numpy.pi / 180.
    d = numpy.array([109, 0.7599, -0.1203, 76.2699, 73.7185, 69.6]) * 1e-3
    theta = numpy.array([-0.2988, 88.8208, 0.0709, -90.3032, -0.0696, 0.0002]) * numpy.pi / 180.
    # 初始化机器人
    my_robot = robot(joint_size, joint_type, a, alpha, d, theta)

    # 定义关节角度
    # q_initial = [0, numpy.pi/4, numpy.pi/4, 0, numpy.pi/4, 0]
    q_initial = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    # 计算末端位姿
    end_effector = my_robot.MFK(q_initial)
    print("m末端位姿:\n", end_effector)

    # # 绘制机器人
    # fig = plt.figure()
    # my_robot.mdraw(q_initial, fig, show_frame=True)
    # # plt.show()

    # 逆运动学示例
    Rt = end_effector[0:3, 0:3]  # 目标姿态
    Pt = end_effector[0:3, 3]    # 目标位置
    Pt[0] += 0.02
    Pt[2] -= 0.02
    
    q_solution = my_robot.MIK(Rt, Pt, q_initial)
    print("逆运动学解:", q_solution)
    end_effector = my_robot.MFK(q_solution)
    print("末端位姿:\n", end_effector)

if __name__ == "__main__":  
    test_eyou_sdh()
    test_eyou_mdh()