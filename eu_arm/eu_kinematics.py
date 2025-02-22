from curi_robotics import robot
from eu_arm_const import *

np.set_printoptions(precision=7, suppress=True)


class eu_arm_kinematics(robot):
    def __init__(self, joint_size=JNT_NUM, joint_type=kJOINT_TYPE, a=kDH_A, alpha=kDH_ALPHA, d=kDH_D, theta=kDH_THETA):
        super().__init__(joint_size, joint_type, a, alpha, d, theta)
        self.name = "eu_arm"
        self.info()

    def info(self):
        print("="*70)
        print("Robot Model: ", self.name)
        print("JOINT_SIZE:  ", self.JOINT_SIZE)
        print("JOINT_TYPE:  ", self.JOINT_TYPE)
        print("A:           ", self.A)
        print("ALPHA:       ", self.ALPHA)
        print("D:           ", self.D)
        print("THETA:       ", self.THETA)
        print("="*70, "\n")