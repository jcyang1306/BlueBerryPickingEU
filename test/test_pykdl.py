from PyKDL import Chain, Joint, Segment, Frame, Rotation, Vector, JntArray, ChainIkSolverPos_LMA, ChainFkSolverPos_recursive, JntSpaceInertiaMatrix 
from PyKDL import *
import numpy as np
from numpy import pi, sin, cos

from benchmark import benchmark


def create_kdl_chain(dh_params):
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

@benchmark
def forward_kinematics(chain, joint_angles):
    fk_solver = ChainFkSolverPos_recursive(chain)
    end_effector_frame = Frame()
    fk_solver.JntToCart(joint_angles, end_effector_frame)
    return end_effector_frame

@benchmark
def inverse_kinematics(chain, q_init, pos_goal, q_sol):
    ik_solver = ChainIkSolverPos_LMA(chain)  # Using Levenberg-Marquardt algorithm
    return ik_solver.CartToJnt(q_init, pos_goal, q_sol)



def deg2rad(deg):
    return deg * pi / 180

def toJntArray(joint_angles):
    joint_array = JntArray(len(joint_angles))
    for i, angle in enumerate(joint_angles):
        joint_array[i] = angle
    return joint_array

def testFKrecursive(chain, joint_angles):
    fk_solver = ChainFkSolverPos_recursive(chain)
    for i in range(1, chain.getNrOfSegments()+1):
        linki_frame = Frame()
        fk_solver.JntToCart(joint_angles, linki_frame, i)
        print(f'link{i} frame\n: {linki_frame}')

if __name__ == "__main__":
    # Define DH parameters for a 6-DOF robot arm
    # [a, alpha, d, theta]
    dh_params = [
        [0.0        , deg2rad(90)      , 109e-3    , deg2rad(-0.2988) ],  # Joint 1
        [183.765e-3 , deg2rad(179.3582), 0.7599e-3 , deg2rad(88.8208) ],  # Joint 2
        [164.6548e-3, deg2rad(179.6965), -0.1203e-3, deg2rad(0.0709)  ],  # Joint 3
        [-0.4715e-3 , deg2rad(-90.2676), 76.2699e-3, deg2rad(-90.3032)],  # Joint 4
        [-1.7394e-3 , deg2rad(90.0063) , 73.7185e-3, deg2rad(-0.0696) ],  # Joint 5
        [0.0        , deg2rad(0)       , 69.6e-3   , deg2rad(0.0002)  ]   # Joint 6
    ]
    chain = create_kdl_chain(dh_params)
    joint_angles = toJntArray([ 2.81955,  0.73928, -0.4205,   0.56949, -1.53532, -0.33057])

    # testFKrecursive(chain, joint_angles)

    # Compute forward kinematics
    end_effector_frame = forward_kinematics(chain, joint_angles)
    print("End-Effector Transformation:\n", end_effector_frame)
    print("End-Effector Position:", end_effector_frame.p)
    print("End-Effector Orientation:", end_effector_frame.M.GetRPY())


    # Inverse kinematics 
    q_goal_ik = toJntArray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    retval = inverse_kinematics(chain, joint_angles, end_effector_frame, q_goal_ik)
    print(f"IK result: [{retval}]", [q_goal_ik[i] for i in range(6)])


    end_effector_frame.p[1] += 0.03
    retval = inverse_kinematics(chain, joint_angles, end_effector_frame, q_goal_ik)
    print(f"IK result: [{retval}]", [q_goal_ik[i] for i in range(6)])
