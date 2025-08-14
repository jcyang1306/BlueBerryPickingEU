import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rm_arm.core.rm_robot_controller import RobotArmController

# [1.57255907312042, 1.09430401061111, -2.3395092533307804, 0.007592182287787225, -0.5523618141880225, 1.8904359782030955]
# [0.00095, -0.23889, 0.501967, -1.809, 0.318, 3.071]


# Realman robot version grasp motion
def async_grasp_v2(marching_dist):
    print("===== stepping forward =====")  

    fwd_motion = np.eye(4)
    fwd_motion[2, 3] = marching_dist
    robot_controller.moveL_relative_v2(fwd_motion, v=50)

    print("===== closing gripper =====")  
    # closeGripper(pump_ctrl)
    # time.sleep(1)

    print("===== rotating =====")  
    q_inc = [0, 0, 0, 0, 0, 1.57]
    robot_controller.moveJ_relative(q_inc, 100, 1)
    
    print("===== getting back =====")  
    bwd_motion = np.eye(4)
    bwd_motion[2, 3] = -0.1
    robot_controller.moveL_relative_v2(bwd_motion, v=100)

def test_get_info():
    robot_controller.get_arm_software_info()
    
    # joint states
    jq = robot_controller.get_joint_state()
    jq_deg = robot_controller.get_joint_state(degrees=True)
    print(f'joint states: {jq}\njoint states (deg): {jq_deg}\n')
    
    # joint states and poses
    curr_q, curr_pose = robot_controller.get_current_state()
    _, pose_mat = robot_controller.get_current_state(fmt='matrix')
    print(f'current q: {curr_q} \ncurrent posevec: {curr_pose} \ncurrent pose matrix: \n{pose_mat}\n')
    
    import ipdb; ipdb.set_trace()
    
    return

def test_moveJ():
    import ipdb; ipdb.set_trace()
    # moveJ
    q_init = [1.57255907312042, 1.09430401061111, -2.3395092533307804, 0.007592182287787225, -0.5523618141880225, 1.8904359782030955]
    robot_controller.moveJ(q_init, v=10)
    
    import ipdb; ipdb.set_trace()
    # moveJ relative, rotate joint 6
    q_inc = [0, 0, 0, 0, 0, 1.57]
    robot_controller.moveJ_relative(q_inc, 70, 1)

    # TODO: add movej_p test
    return

def test_follow_offline_traj():
    return

def test_moveL():
    pose_init = [0.00095, -0.23889, 0.501967, -1.809, 0.318, 3.071]
    robot_controller.moveL(pose_init, v=50)

    # movel relative, not worked
    # fwd_motion = np.eye(4)
    # fwd_motion[2, 3] = 0.1
    # robot_controller.moveL_relative_v2(fwd_motion, v=50)

    import ipdb; ipdb.set_trace()
    # movel relative, not worked
    # fwd_motion =[0, 0, 0, 0, 0, 0]
    # ret = robot_controller.moveL_relative(fwd_motion)
    test_flag=0
    
    import ipdb; ipdb.set_trace()
    # move_to_pose
    # _, curr_pose = robot_controller.get_current_state(fmt='matrix')
    # curr_pose[2, 3] += 0.05
    # robot_controller.move_to_pose(curr_pose)

if __name__ == '__main__':
    np.set_printoptions(precision=7, suppress=True)
    import time
    
    robot_controller = RobotArmController("192.168.1.18", 8080, 3)
    ret = robot_controller.robot.rm_change_work_frame("Base")
    robot_controller.get_arm_software_info()
    
    
    ##===========forward kinematics=========
    jqs = [[0, 0, 0, 0, 0, 0.],
           [90,90,90,90, 90,90],
           [90,90,90,90, 90, 0],
           [90,90,90,60, 0, 0],
           [90,90,90, 0, 0, 0],
           [90,90, 0.,0, 0, 0],
           [90, 0, 0.,0, 0, 0],
           [1.57255907312042, 1.09430401061111, -2.3395092533307804, 0.007592182287787225, -0.5523618141880225, 1.8904359782030955]
           
    ]
    for jq in jqs:
        pose = robot_controller.robot.rm_algo_forward_kinematics(jq, 0)
        print(f'\njq: {np.array(jq)}')
        print(f'pose: {np.array(pose)}')
    
    ##===========forward kinematics=========
    

    import ipdb; ipdb.set_trace()
    test_get_info()
    
    # import ipdb; ipdb.set_trace()
    # test_moveJ()
    
    # import ipdb; ipdb.set_trace()
    # test_moveL()
    
    # import ipdb; ipdb.set_trace()
    # t0 = time.time()
    # async_grasp_v2(0.1)
    # print(f'time: {time.time()-t0}')

    print('done')