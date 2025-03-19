import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as Rot
import copy, csv
import threading

from dev.rs_d405 import RealSenseController
from dev.pump_control import PumpControl
from utils.circular_linked_list import CLinkedList

from eu_arm.eu_arm_interface import *

# TODO: resolve include path issue
from grasp_algo import GraspingAlgo

np.set_printoptions(precision=7, suppress=True)

import time
import numpy as np

use_pump = True
JOINTLIMIT = 3.14

    # [-0.9996523,  0.025658 ,  0.0060854,  0.0081949],
    # [-0.0256909, -0.9996554, -0.0053882,  0.0506086],
    # [ 0.0059451, -0.0055427,  0.999967 ,  0.0800145],
    # [ 0.       ,  0.       ,  0.       ,  1.       ]


H_handeye = np.array([
    [-0.9996523,  0.025658 ,  0.0060854,  0.0081949],
    [-0.0256909, -0.9996554, -0.0053882,  0.0476086],
    [ 0.0059451, -0.0055427,  0.999967 ,  0.0800145],
    [ 0.       ,  0.       ,  0.       ,  1.       ]
]) 

eef_offset = np.array(
[[1.0,  0.0,  0.0, 0.0  ],
 [0.0 , 1.0,  0.0, 0.0  ],
 [0.0 ,  0.0,  1.0, -0.14],
 [0.  ,  0. ,  0. ,  1.  ]])

EEF_STEP = 0.0015

eef_step_fwd = np.eye(4)
eef_step_fwd[2, 3] = EEF_STEP

eef_step_bwd = np.eye(4)
eef_step_bwd[2, 3] = -EEF_STEP

# hardcoded object pose, ensures topdown
detected_obj_pose = np.array([
    [-0.9990327,  0.0428815, -0.009739 ,  0.0446881],
    [-0.0358097, -0.9218934, -0.3857852, -0.0301231],
    [-0.0255213, -0.3850633,  0.9225372,  0.2100264],
    [ 0.       ,  0.       ,  0.       ,  1.       ]])


# hardcoded object pose, ensures topdown
detected_obj_pose_viewpoint = np.array([
    # [-1.0,  0.0, 0.0 ,  0.0],
    # [0.0 , -1.0, 0.0 ,  0.0],
    # [0.0 ,  0.0, 1.0 ,  0.1],
    # [ 0. ,  0. ,  0. ,  1.       ]])
    [-0.9693427,  0.2454546, -0.0112599,  0.0],
    [-0.1902715, -0.778833 , -0.5976754,  0.0],
    [-0.1554718, -0.5772098,  0.8016591,  0.1],
    [ 0. ,  0. ,  0. ,  1.       ]])

# @benchmark
def infer(algo, img):
    return algo.infer_img(img)

# @benchmark
def servoL(relative_pos):
    # robot arm fk
    q_curr = eu_get_current_joint_positions()
    eef_pose_frame = eu_arm.FK(q_curr)
    eef_pose = eu_arm.frame2mat(eef_pose_frame)
    
    grasp_pose = eef_pose @ relative_pos
    grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
    
    ec = 0 # NoError
    q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
    print(f'grasp js pose: {q_grasp}')
    print(f'delta q: {q_curr - q_grasp}')
    
    if np.max(np.max(q_grasp)) > JOINTLIMIT:
        print('\n===== joint out of range <exceed joint limit> =====')
    elif np.max(np.abs(q_curr - q_grasp) > 0.075):
        print('\n===== joint out of range <delta q> =====')
    else:
        eu_mov_to_target_jnt_pos(q_grasp)

def moveL(z_offset):
    eu_set_joint_velocities([1,1,1,1,1,1])
    stps_total = int(np.abs(z_offset)/EEF_STEP)
    if z_offset > 0:
        eef_step_mat = eef_step_fwd
    else:
        eef_step_mat = eef_step_bwd
    for i in range(stps_total):
        servoL(eef_step_mat)
        time.sleep(0.01)
    print(f"============moveL done [{stps_total}] ============")

def moveRelativeAsync(z_offset):
    eu_set_joint_velocities([1,1,1,1,1,10])
    t = threading.Thread(target=moveL, args=(z_offset, ))
    t.start()

def closeGripper(pump_ctrl):
    if pump_ctrl is not None:
        pump_ctrl.config_gripper(1)

def releaseGripper(pump_ctrl):
    if pump_ctrl is not None:
        pump_ctrl.config_gripper(0)

if __name__ == "__main__":  
    # EU Robot Arm initialization 
    eu_arm = eu_arm_kinematics()
    if eu_init_device(1000):
        print("eu_init_device ok")
        if eu_get_heartbeat():
            print("eu_get_heartbeat ok")

            eu_enable_motor(True)
            eu_set_joint_max_velocity(10)
            eu_set_joint_velocities([2,2,3,5,2,10])
            eu_set_work_mode(ControlMode.POSITION_MODE)
            
            # move to initial pose            
            # b2 [-1.3804397, -0.5810595,  0.6262863,  0.5990024,  1.5148448, -1.1884477]
            # Desktop [-0.04506, -0.20009,  1.14981,  0.04487,  1.57482,  0.19491]

            # q_init = np.array([-1.5678242,  0.5048714,  1.8200682,  1.1766591,  1.644715 , -1.3820208]) #  demo init pose1
            
            # q_init = np.array([-0.3098641, -0.2427525,  1.5173946,  1.494289 ,  0.4976724, -1.1005813]) #   demo init pose2
            q_init = np.array([-1.2384019,  0.0845607,  1.7863206, -1.6458655, -1.1614152, 1.4325463]) #  

            eu_mov_to_target_jnt_pos(q_init)
            # q_init = eu_get_current_joint_positions()

    # Pump init
    if use_pump:
        pump_ctrl = PumpControl()
    else:
        pump_ctrl = None

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

            if key & 0xFF == ord('s'):
                print("\n===== start inference =====")
                t0 = time.time()
                masks = infer(grasp_algo, color_img)
                print(f'>>> infer {time.time()-t0}')

                if masks is not None and len(masks) > 0:
                    pc = rs_ctrl.convert_to_pointcloud(color_frame, depth_frame)
                    grasp_poses = grasp_algo.gen_grasp_pose(color_img, pc, masks)
                    print(f'grasppose gen {time.time()-t0}')
                    if grasp_poses is not None:                        
                        first_obj_pose = grasp_poses[0]
                        detected_obj_pose_viewpoint[:3,3] = first_obj_pose[:3]
                        
                        print(f"selected object pose: \n{repr(first_obj_pose)}")
                        print(f'================\ndetected pose: \n{repr(detected_obj_pose_viewpoint)}\n====================')

                        eef2obj = H_handeye @ detected_obj_pose_viewpoint 
                        print(f'eef2obj: \n{repr(eef2obj)}')
                        
                else:
                    print('\n===== no object detected =====')
                continue

            if key & 0xFF == ord('g'):
                print("\n go grasping")  
                q_curr = eu_get_current_joint_positions()
                eef_pose = eu_arm.frame2mat(eu_arm.FK(q_curr))

                # pregrasp pose -> relative pose on x-y plane wrt eef
                eef2obj = H_handeye @ detected_obj_pose_viewpoint 
                eef2pre_grasp= np.eye(4)
                eef2pre_grasp[:3, 3] = eef2obj[:3, 3]
                if eef2obj[2, 3] - 0.14 > 0.075:
                    eef2pre_grasp[2, 3] = 0.075
                else:
                    eef2pre_grasp[2, 3] = eef2obj[2, 3] - 0.14
                print(f'marching_dist: {eef2obj[2, 3] - 0.14} | pre_grasp_dist: [{eef2pre_grasp[2, 3]}]')
                print(f'eef2pre_grasp: \n{repr(eef2pre_grasp)}')

                H_base_to_obj = eef_pose @ eef2pre_grasp
                grasp_pose_tcp = H_base_to_obj

                print(f'eef_pose: \n{repr(eef_pose)}')
                print(f'H_base_to_obj: \n{repr(H_base_to_obj)}')
                print(f'grasp_pose: \n{repr(grasp_pose_tcp)}')

                grasp_pose_frame = eu_arm.mat2frame(grasp_pose_tcp)
                ec = 0 # NoError
                q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
                print(f'curr  js pose: {q_curr}')
                print(f'grasp js pose: {q_grasp}')

                if np.max(np.max(q_grasp)) > 2.8:
                    print('\n===== joint out of range =====')
                else:
                    eu_mov_to_target_jnt_pos(q_grasp)

                time.sleep(3)
                continue

            if key & 0xFF == ord('t'):
                print("===== stepping down =====")  
                moveRelativeAsync(0.20)
                time.sleep(2.5)

                print("===== closing gripper =====")  
                closeGripper(pump_ctrl)
                time.sleep(1)
                q_target = eu_get_current_joint_positions()
                q_target[5] += 90 / 180 * np.pi # 90deg
                eu_set_joint_velocities([2,2,3,5,2,10])
                
                eu_mov_to_target_jnt_pos(q_target)
                time.sleep(3)

                print("===== stepping up =====")  
                moveRelativeAsync(-0.38)
                time.sleep(2)

                print("===== releasing gripper =====")  
                # releaseGripper(pump_ctrl)
                continue

            if key & 0xFF == ord('m'):
                print("\n moving along eef z axis")  
                moveRelativeAsync(0.03)
                continue

            if key & 0xFF == ord('u'):
                print("\n moving along eef z axis")  
                moveRelativeAsync(-0.03)
                continue

            if key & 0xFF == ord('c'):
                print("\n===== closing gripper =====")  
                closeGripper(pump_ctrl)
                continue

            if key & 0xFF == ord('r'):
                print("\n===== releasing gripper =====")  
                releaseGripper(pump_ctrl)
                continue

            if key & 0xFF == ord('i'):
                # q_init = np.array([-0.04506, -0.20009,  1.14981,  0.04487,  1.57482,  0.19491])
                eu_set_joint_velocities([2,2,3,5,2,10])
                eu_mov_to_target_jnt_pos(q_init)
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