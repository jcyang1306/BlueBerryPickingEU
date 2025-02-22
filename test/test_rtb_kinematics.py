from numpy import pi, sin, cos
import numpy as np

from eu_arm.eu_arm_interface import *

import roboticstoolbox as rtb
from spatialmath import *

np.set_printoptions(precision=7, suppress=True)

def deg2rad(deg):
    return deg * pi / 180

def rad2deg(rad):
    return rad * 180 / pi

class eu_arm_rtb(rtb.DHRobot):
    def __init__(self):
        super().__init__(
            [
            rtb.RevoluteDH(a=0,           alpha=deg2rad(90),       d=109e-3    , offset=deg2rad(-0.2988) ),
            rtb.RevoluteDH(a=183.765e-3,  alpha=deg2rad(179.3582), d=0.7599e-3 , offset=deg2rad(88.8208) ),
            rtb.RevoluteDH(a=164.6548e-3, alpha=deg2rad(179.6965), d=-0.1203e-3, offset=deg2rad(0.0709)  ),
            rtb.RevoluteDH(a=-0.4715e-3,  alpha=deg2rad(-90.2676), d=76.2699e-3, offset=deg2rad(-90.3032)),
            rtb.RevoluteDH(a=-1.7394e-3,  alpha=deg2rad(90.0063),  d=73.7185e-3, offset=deg2rad(-0.0696) ),
            rtb.RevoluteDH(a=0,           alpha=deg2rad(0),        d=69.6e-3   , offset=deg2rad(0.0002) )
            ], name='eu_arm'
        )

if __name__ == '__main__':
    eu_arm_kin = eu_arm_kinematics()
    eu_arm_rtb = eu_arm_rtb()
    q0 = np.array([ 2.81955,  0.73928, -0.4205,   0.56949, -1.53532, -0.33057] )
    print(f'q0: {repr(q0)}')

    t0=time.time()
    eef_pose_kin = eu_arm_kin.MFK(q0)
    print(f'kin: {repr(eef_pose_kin)}, time: {time.time()-t0}')

    # t0=time.time()
    # eef_pose_rtb = eu_arm_rtb.fkine(q0).A
    # print(f'rtb: {repr(eef_pose_rtb)}, time: {time.time()-t0}')


    # eef_pose_kin[1,3] += 0.03
    # eef_pose_rtb[1,3] += 0.03

    # t0=time.time()
    # q1_kin = eu_arm_kin.MIK_from_T(eef_pose_kin, q0)
    # print(f'q kin: {repr(q1_kin)}, time: {time.time()-t0}')

    # t0=time.time()
    # q1_rtb = eu_arm_rtb.ikine_NR(eef_pose_rtb, q0=q0).q
    # print(f'q rtb: {repr(q1_rtb)}, time: {time.time()-t0}')

    # print(f'kin: {repr(eu_arm_kin.MFK(q1_kin))}')
    # print(f'rtb: {repr(eu_arm_rtb.fkine(q1_rtb).A)}')

    # plan_joint_space(q0, q1_kin)
