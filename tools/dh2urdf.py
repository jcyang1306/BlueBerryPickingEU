import numpy as np
from eu_arm.eu_arm_const import kJNT_NUM, kJOINT_TYPE, kDH_A, kDH_ALPHA, kDH_D, kDH_THETA # TODO: fix format mismatch with KDL 
from scipy.spatial.transform import Rotation as Rot

def deg2rad(deg):
    return deg * np.pi / 180

def rad2deg(rad):
    return rad * 180 / np.pi

def Zi(theta, d):
    return np.array([[+np.cos(theta), -np.sin(theta), 0, 0],
                    [+np.sin(theta), +np.cos(theta), 0, 0],
                    [             0,              0, 1, d],
                    [             0,              0, 0, 1]])

def Xi(alpha, a):
    return np.array([[1,              0,              0, a],
                    [0, +np.cos(alpha), -np.sin(alpha), 0],
                    [0, +np.sin(alpha), +np.cos(alpha), 0],
                    [0,              0,              0, 1]])


np.set_printoptions(precision=7, suppress=True)

if __name__ == "__main__":
    # initial T matrix and joint angles
    T = np.eye(4)
    joint_angles = [0., 0., 0., 0., 0., 0.]


    for i in range(6):
        print(f"\nJoint {i+1} angle: {joint_angles[i]}")
        JntA = Zi(deg2rad(kDH_THETA[i]), kDH_D[i]) @ Xi(deg2rad(kDH_ALPHA[i]), kDH_A[i])
        T = T @ JntA
        rot = JntA[:3,:3]
        rpy = Rot.from_matrix(rot).as_euler('xyz', degrees=True)
        print(f"Jnt{i+1}, {repr(JntA)}")
        print(f"Jnt{i+1}, <origin> xyz= {JntA[:3,3]}, rpy={deg2rad(rpy)}\n")

    print(f"\n\n tcp pose: {repr(T)}")