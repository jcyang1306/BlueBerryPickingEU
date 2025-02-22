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
            rtb.RevoluteDH(a=0,           alpha=deg2rad(90),       d=109e-3, offset=deg2rad(-0.2988)),
            rtb.RevoluteDH(a=183.765e-3,  alpha=deg2rad(179.3582), d=0.7599e-3, offset=deg2rad(88.8208)),
            rtb.RevoluteDH(a=164.6548e-3, alpha=deg2rad(179.6965), d=-0.1203e-3, offset=deg2rad(0.0709)),
            rtb.RevoluteDH(a=-0.4715e-3,  alpha=deg2rad(-90.2676), d=76.2699e-3, offset=deg2rad(-90.3032)),
            rtb.RevoluteDH(a=-1.7394e-3,  alpha=deg2rad(90.0063),  d=73.7185e-3, offset=deg2rad(-0.0696)),
            rtb.RevoluteDH(a=0,           alpha=deg2rad(0),       d=69.6e-3, offset=deg2rad(0.0002))
            ], name='eu_arm'
        )

if __name__ == '__main__':
    eu_arm = eu_arm_interface()

    q0 = np.array([1.56955 , -0.7202197, -1.3903205, -0.0901917, -1.570317 , -0.0003835] )
    print(f'q0: {repr(q0)}')
    jmove_blocking(q0)

    t0=time.time()
    eef_pose_kin = eu_arm.computeFK(q0)
    print(f'kin: {repr(eef_pose_kin)}\n time: {time.time()-t0}')

    eef_pose_kin[1,3] += 0.03
    eef_pose_kin[0,3] -= 0.08

    t0=time.time()
    q1_kin = eu_arm.computeIK(eef_pose_kin, q0)
    print(f'q kin: {repr(q1_kin)}\n time: {time.time()-t0}')

    t0=time.time()
    # q_traj = eu_arm.move_relative(np.array([-0.10, -0.06, 0.03]).T)
    # print(f'>>>> planning time: {time.time()-t0}')

    q_traj = plan_joint_space(q0, q1_kin, 50)

    # t0 = time.time()
    # for q in q_traj:
    #     print(f'q: {q}')
    #     eu_servoJ(q)
    #     time.sleep(0.03)
    # print(f'---Avg control freq [{len(q_traj) / (time.time()-t0)}]---')
    
    # time.sleep(2)
    # print(eu_arm.get_TCP_pose())

    eu_arm.move_relative_ruckig(np.array([-0.3, -0.03, -0.03]).T)