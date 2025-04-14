from PyKDL import Chain, Joint, Segment, Frame, Rotation, Vector, JntArray, ChainIkSolverPos_LMA, ChainFkSolverPos_recursive, JntSpaceInertiaMatrix 
from PyKDL import *
# from scipy.spatial.transform import Rotation as R

import numpy as np
from numpy import pi, sin, cos, deg2rad, rad2deg

from utils.benchmark import benchmark
class KDLKinematics():
    def __init__(self, joint_size, joint_type, a, alpha, d, theta):
        self.JOINT_SIZE = joint_size
        self.A = a
        self.ALPHA = alpha
        self.D = d
        self.THETA = theta
        if joint_type == []:
            self.JOINT_TYPE = np.zeros((joint_size))
        else:
            self.JOINT_TYPE = np.array(joint_type)

        # KDL stuff
        dh_params = [
            [a, deg2rad(alpha), d, deg2rad(theta)]
            for a, alpha, d, theta in zip(self.A, self.ALPHA, self.D, self.THETA)
        ]
        self.chain = self.create_kdl_chain(dh_params)
        self.fk_pos_solver = ChainFkSolverPos_recursive(self.chain)
        self.ik_pos_solver = ChainIkSolverPos_LMA(self.chain) # Using Levenberg-Marquardt algorithm
        self.init_ = True # TODO: check if init succeed
        return

    def create_kdl_chain(self, dh_params):
        chain = Chain()
        idx = 0
        for a, alpha, d, theta in dh_params:
            print(f'Jnt[{idx}] | a: {a:.7f}, alpha: {alpha:.7f}, d: {d:.7f}, theta: {theta:.7f}')
            joint = Joint(Joint.RotZ)  # Assuming rotational joints around Z-axis
            frame = Frame.DH(a, alpha, d, theta) # TODO: ? suppose to be MDH??????
            chain.addSegment(Segment(joint, frame))
            idx += 1
        print(f'Create chain success with [{chain.getNrOfSegments()}] segments')
        return chain

    def toJntArray(self, joint_angles):
        joint_array = JntArray(len(joint_angles))
        for i, angle in enumerate(joint_angles):
            joint_array[i] = angle
        return joint_array

    # @benchmark
    def FK(self, q):
        return self.forward_kinematics(q)

    # @benchmark
    def IK(self, q_init, pos_goal, retval):
        # TODO: add solution limit checking
        q_sol = JntArray(len(q_init))
        retval = self.ik_pos_solver.CartToJnt(self.toJntArray(q_init), pos_goal, q_sol) 
        q_sol = [q_sol[i] for i in range(len(q_init))]
        return q_sol

    def computeFK(self, q):
        pose_frame = self.forward_kinematics(q)
        return self.frame2mat(pose_frame)
    
    def computeIK(self, q_init, pos_goal, retval):
        pose_frame = self.mat2frame(pos_goal)
        return self.IK(q_init, pose_frame, retval)

    def forward_kinematics(self, q):
        end_effector_frame = Frame()
        joint_angles = self.toJntArray(q)
        self.fk_pos_solver.JntToCart(joint_angles, end_effector_frame)
        return end_effector_frame

    # @benchmark
    def frame2mat(self, frame):
        rotx = np.array(list(frame.M.UnitX())).reshape(3,1)
        roty = np.array(list(frame.M.UnitY())).reshape(3,1)
        rotz = np.array(list(frame.M.UnitZ())).reshape(3,1)
        tvec = np.array(list(frame.p)).reshape(3,1)
        hmat = np.hstack([rotx, roty, rotz, tvec])
        return np.vstack([hmat, np.array([0,0,0,1])])
    
    def mat2frame(self, mat):
        rot = Rotation(mat[0, 0], mat[0, 1], mat[0, 2],
                       mat[1, 0], mat[1, 1], mat[1, 2],
                       mat[2, 0], mat[2, 1], mat[2, 2])
        tvec = Vector(mat[0, 3], mat[1, 3], mat[2, 3])
        return Frame(rot, tvec)

if __name__ == "__main__":
    # Define DH parameters for a 6-DOF robot arm
    # [a, alpha, d, theta]
    # dh_params = [
    #     [0.0        , 90      , 109e-3    , -0.2988 ],  # Joint 1
    #     [183.765e-3 , 179.3582, 0.7599e-3 , 88.8208 ],  # Joint 2
    #     [164.6548e-3, 179.6965, -0.1203e-3, 0.0709  ],  # Joint 3
    #     [-0.4715e-3 , -90.2676, 76.2699e-3, -90.3032],  # Joint 4
    #     [-1.7394e-3 , 90.0063 , 73.7185e-3, -0.0696 ],  # Joint 5
    #     [0.0        , 0       , 69.6e-3   , 0.0002  ]   # Joint 6
    # ]
    from eu_arm.eu_arm_const import kJNT_NUM, kJOINT_TYPE, kDH_A, kDH_ALPHA, kDH_D, kDH_THETA # TODO: fix format mismatch with KDL 

    kdl_kin = KDLKinematics(kJNT_NUM, kJOINT_TYPE, kDH_A, kDH_ALPHA, kDH_D, kDH_THETA)
    joint_angles = [-0.0450607, -0.2001845, 1.1496227, 0.0447731, 1.5747272, 0.1947197]

    # Compute forward kinematics
    end_effector_frame = kdl_kin.FK(joint_angles)
    print("End-Effector Transformation:\n", end_effector_frame)
    print("End-Effector Position:", end_effector_frame.p)
    print("End-Effector Orientation:", end_effector_frame.M.GetRPY())

    # To Transformation matrix
    eef_pose = kdl_kin.frame2mat(end_effector_frame)
    print(f"End-Effector Transformation: \n{eef_pose}")
    
    back_frame = kdl_kin.mat2frame(eef_pose)
    print(f"Back conversion frame: \n{back_frame}")



    # Inverse kinematics 
    errcode = 0 # no error
    q_goal_ik = kdl_kin.IK(joint_angles, end_effector_frame, errcode)
    print(f"IK result: [{errcode}]", [q_goal_ik[i] for i in range(6)])

    end_effector_frame.p[1] += 0.03
    q_goal_ik = kdl_kin.IK(joint_angles, end_effector_frame, errcode)
    print(f"IK result with pose offset: [{errcode}]", [q_goal_ik[i] for i in range(6)])
