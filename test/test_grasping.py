import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as Rot
import copy, csv

from dev.rs_d405 import RealSenseController
from dev.pump_control import PumpControl
from utils.circular_linked_list import CLinkedList

from eu_arm.eu_arm_interface import *

# TODO: resolve include path issue
from grasp_algo import GraspingAlgo

np.set_printoptions(precision=7, suppress=True)

import time
import numpy as np

H_handeye = np.array([
    [-0.9693427, -0.1902716, -0.1554718,  0.0225967],
    [ 0.2454546, -0.778833 , -0.5772098,  0.049451 ],
    [-0.01126  , -0.5976754,  0.8016591,  0.088671 ],
    [ 0.       ,  0.       ,  0.       ,  1.       ]
    ]) 


eef_offset = np.array(
[[1.0,  0.0,  0.0, 0.0  ],
 [0.0 , 1.0,  0.0, 0.0  ],
 [0.0 ,  0.0,  1.0, -0.14],
 [0.  ,  0. ,  0. ,  1.  ]])

eef_step_fwd = np.eye(4)
eef_step_fwd[2, 3] = 0.0015

eef_step_bwd = np.eye(4)
eef_step_bwd[2, 3] = -0.0015

detected_obj_pose = np.array([[-0.9990327,  0.0428815, -0.009739 ,  0.0446881],
       [-0.0358097, -0.9218934, -0.3857852, -0.0301231],
       [-0.0255213, -0.3850633,  0.9225372,  0.2100264],
       [ 0.       ,  0.       ,  0.       ,  1.       ]])

@benchmark
def infer(algo, img):
    return algo.infer_img(img)

if __name__ == "__main__":  
    # EU Robot Arm initialization 
    eu_arm = eu_arm_kinematics()
    if eu_init_device(1000):
        print("eu_init_device ok")
        if eu_get_heartbeat():
            print("eu_get_heartbeat ok")

            eu_enable_motor(True)
            eu_set_joint_max_velocity(3)
            eu_set_joint_velocities([2,2,3,5,2,3])
            eu_set_work_mode(ControlMode.POSITION_MODE)
            
            # move to initial pose            
            q_init = np.array([-0.04506, -0.20009,  1.14981,  0.04487,  1.57482,  0.19491])
            eu_mov_to_target_jnt_pos(q_init)
            # q_init = eu_get_current_joint_positions()

    # Pump init
    pump_ctrl = PumpControl()

    # Realsense init
    rs_ctrl = RealSenseController()
    rs_ctrl.config_stereo_stream()
    cam_intr, distor_coeff = rs_ctrl.get_intrinsics()
    # rs_ctrl.start_streaming()


    # Vision Algo
    grasp_algo = GraspingAlgo()

    # Visualization
    cv2.namedWindow('rs_calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rs_calibration', int(1920/2), int(1080/2))

    ## ================= Global Variables =================
    # EU arm control stuff
    j_idx = CLinkedList([1,2,3,4,5,6]).head
    q_curr = eu_get_current_joint_positions()
    JMOVE_STEP = 0.005
    q_target = q_init

    ## ================= Global Variables =================

    import time

    try:
        while True:
            start = time.time()

            # # Update robot states
            # q_curr = eu_get_current_joint_positions()
            color_frame, depth_frame = rs_ctrl.get_stereo_frame()
            color_img = np.asanyarray(color_frame.get_data())

            # Visiualization and interactive control
            cv2.imshow('rs_calibration', color_img)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27: # Press esc or 'q' to close the image window
                print("\n===== stop pipeline =====")  
                cv2.destroyAllWindows()
                break

            if key & 0xFF == ord('c'):
                print("\n===== closing gripper =====")  
                pump_ctrl.config_gripper(1)
                continue

            if key & 0xFF == ord('r'):
                print("\n===== releasing gripper =====")  
                pump_ctrl.config_gripper(0)
                continue

            if key & 0xFF == ord('s'):
                masks = infer(grasp_algo, color_img)
                if masks is not None:
                    pc = rs_ctrl.convert_to_pointcloud(color_frame, depth_frame)
                    grasp_poses = grasp_algo.gen_grasp_pose(color_img, pc, masks)
                    if grasp_poses is not None:                        
                        first_obj_pose = grasp_poses[0]
                        detected_obj_pose[:3,3] = first_obj_pose[:3]
                        print(f"selected object pose: \n{repr(first_obj_pose)}")
                        print(f'================\ndetected pose: \n{repr(detected_obj_pose)}\n====================')

                else:
                    print('\n===== no object detected =====')
                continue


            if key & 0xFF == ord('g'):
                print("\n go grasping")  
                q_curr = eu_get_current_joint_positions()
                eef_pose = eu_arm.frame2mat(eu_arm.FK(q_curr))

                H_base_to_obj = eef_pose @ H_handeye @ detected_obj_pose 
                grasp_pose_tcp = H_base_to_obj @ eef_offset

                print(f'H_base_to_obj: \n{repr(H_base_to_obj)}')
                print(f'grasp_pose: \n{repr(grasp_pose_tcp)}')

                grasp_pose_frame = eu_arm.mat2frame(grasp_pose_tcp)
                ec = 0 # NoError
                q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
                print(f'curr  js pose: {q_curr}')
                print(f'grasp js pose: {q_grasp}')

                if np.max(np.max(q_grasp)) > 1.6:
                    print('\n===== joint out of range =====')
                else:
                    eu_mov_to_target_jnt_pos(q_grasp)

                time.sleep(3)
                continue


            if key & 0xFF == ord('m'):
                print("\n moving along eef z axis")  

                # robot arm fk
                q_curr = eu_get_current_joint_positions()
                eef_pose_frame = eu_arm.FK(q_curr)
                eef_pose = eu_arm.frame2mat(eef_pose_frame)
                print(f'eef_pose: \n{repr(eef_pose)}')

                grasp_pose = eef_pose @ eef_step_fwd
                print(f'grasp_pose: \n{repr(grasp_pose)}')

                grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
                ec = 0 # NoError
                q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
                print(f'grasp js pose: {q_grasp}')
                print(f'delta q: {q_curr - q_grasp}')
                
                if np.max(np.max(q_grasp)) > 1.6 or np.max(np.abs(q_curr - q_grasp) > 0.075):
                    print('\n===== joint out of range =====')
                else:
                    eu_set_joint_velocities([1,1,1,1,1,1])
                    eu_mov_to_target_jnt_pos(q_grasp)
                continue

            if key & 0xFF == ord('u'):
                print("\n moving along eef z axis")  

                # robot arm fk
                q_curr = eu_get_current_joint_positions()
                eef_pose_frame = eu_arm.FK(q_curr)
                eef_pose = eu_arm.frame2mat(eef_pose_frame)
                print(f'eef_pose: \n{repr(eef_pose)}')

                grasp_pose = eef_pose @ eef_step_bwd
                print(f'grasp_pose: \n{repr(grasp_pose)}')

                grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
                ec = 0 # NoError
                q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
                print(f'grasp js pose: {q_grasp}')
                print(f'delta q: {q_curr - q_grasp}')
                
                if np.max(np.max(q_grasp)) > 1.6 or np.max(np.abs(q_curr - q_grasp) > 0.075):
                    print('\n===== joint out of range =====')
                else:
                    eu_set_joint_velocities([1,1,1,1,1,1])
                    eu_mov_to_target_jnt_pos(q_grasp)
                continue

            if key & 0xFF == 82: # up 
                q_target[j_idx.data-1] += JMOVE_STEP
                eu_mov_to_target_jnt_pos(q_target)
                continue
            if key & 0xFF == 84: # down
                q_target[j_idx.data-1] -= JMOVE_STEP
                eu_mov_to_target_jnt_pos(q_target)
                continue
            if key & 0xFF == 81: # left 
                print("\n===== switch to next joint =====")  
                j_idx = j_idx.prev
                continue
            if key & 0xFF == 83: # right
                print("\n===== switch to next joint =====")  
                j_idx = j_idx.next
                continue
    finally:
        rs_ctrl.stop_streaming()