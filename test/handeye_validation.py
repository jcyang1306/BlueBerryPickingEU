import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as Rot
import copy, csv

from dev.rs_d405 import RealSenseController
from dev.pump_control import PumpControl
from utils.circular_linked_list import CLinkedList

from eu_arm.eu_arm_interface import *

np.set_printoptions(precision=7, suppress=True)

import time
import numpy as np


# Define chessboard corners, return corner points
def make_chessboard_corners(w=11, h=8, corner_size_mm=0.006):
    # Construct chessboard corners matrix
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp*corner_size_mm  # mm
    return objp


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

eef_step_down = np.eye(4)
eef_step_down[2, 3] = 0.0015

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
    rs_ctrl.config_streaming_color()
    cam_intr, distor_coeff = rs_ctrl.get_intrinsics()
    print(f'cam_intr: {repr(cam_intr)}, distcoeff: {distor_coeff}')
    # rs_ctrl.start_streaming()

    corners_pts = make_chessboard_corners()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001) # threshold 0.001


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

            # Find chessboard corners   
            color_img = rs_ctrl.get_color_frame()
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            ret_find_corners, corners = cv2.findChessboardCorners(gray, (11,8),None) # TODO: fix hardcode h,w
            if ret_find_corners == True:
                # refine corner to subpixel precision
                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(color_img, (11,8), corners, ret_find_corners)

            ## Logging
            print(f"\ Joint [{j_idx.data}] selected, \ntarget_q: {repr(q_target)} \n--- {1.0/(time.time() - start)} fps\n")  

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

            if key & 0xFF == ord('r'):
                print("\n===== releasing gripper =====")  
                pump_ctrl.config_gripper(0)

            if key & 0xFF == ord('m'):
                print("\n moving along eef z axis")  

                # robot arm fk
                q_curr = eu_get_current_joint_positions()
                eef_pose_frame = eu_arm.FK(q_curr)
                eef_pose = eu_arm.frame2mat(eef_pose_frame)
                print(f'eef_pose: \n{repr(eef_pose)}')

                grasp_pose = eef_pose @ eef_step_down
                print(f'grasp_pose: \n{repr(grasp_pose)}')

                grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
                ec = 0 # NoError
                q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
                print(f'grasp js pose: {q_grasp}')
                print(f'delta q: {q_curr - q_grasp}')
                
                if np.max(np.max(q_grasp)) > 1.6 or np.max(np.abs(q_curr - q_grasp) > 0.05):
                    print('\n===== joint out of range =====')
                else:
                    eu_set_joint_velocities([1,1,1,1,1,1])
                    eu_mov_to_target_jnt_pos(q_grasp)
                continue

            if key & 0xFF == ord('g'):
                print("\n go grasping")  
                ret_pnp, rvec, tvec = cv2.solvePnP(corners_pts, corners, cam_intr, distor_coeff)
                if ret_pnp == True:
                    H_cam2obj = np.hstack(((cv2.Rodrigues(rvec))[0], tvec))
                    H_cam2obj = np.vstack((H_cam2obj, np.array([0, 0, 0, 1])))
                    print(f'Hmat pnp [{repr(H_cam2obj)}]')

                # robot arm fk
                q_curr = eu_get_current_joint_positions()
                eef_pose_frame = eu_arm.FK(q_curr)
                eef_pose = eu_arm.frame2mat(eef_pose_frame)
                print(f'eef_pose: \n{repr(eef_pose)}')

                H_base_to_obj = eef_pose @ H_handeye @ H_cam2obj 
                grasp_pose = H_base_to_obj @ eef_offset
                print(f'H_base_to_obj: \n{repr(H_base_to_obj)}')
                print(f'grasp_pose: \n{repr(grasp_pose)}')

                grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
                ec = 0 # NoError
                q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
                print(f'curr  js pose: {q_curr}')
                print(f'grasp js pose: {q_grasp}')

                if np.max(np.max(q_grasp)) > 1.6:
                    print('\n===== joint out of range =====')
                else:
                    eu_mov_to_target_jnt_pos(q_grasp)

                time.sleep(3)
                break

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