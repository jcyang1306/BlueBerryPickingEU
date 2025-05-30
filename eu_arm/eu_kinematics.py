import numpy as np
from eu_arm.eu_arm_const import kJNT_NUM, kJOINT_TYPE, kDH_A, kDH_ALPHA, kDH_D, kDH_THETA # TODO: fix format mismatch with KDL 
from eu_arm.kdl_robot_kinematics import KDLKinematics
np.set_printoptions(precision=7, suppress=True)
class eu_arm_kinematics(KDLKinematics):
    def __init__(self, joint_size=kJNT_NUM, joint_type=kJOINT_TYPE, a=kDH_A, alpha=kDH_ALPHA, d=kDH_D, theta=kDH_THETA):
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