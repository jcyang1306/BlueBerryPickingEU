import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as Rot
import copy, csv, datetime
import time

import open3d as o3d

from dev.rs_d405 import RealSenseController
from dev.pump_control import PumpControl
from utils.circular_linked_list import CLinkedList

from eu_arm.eu_arm_interface import *
from eu_arm.common import *
from env_cfg import *

# TODO: resolve include path issue
from grasp_algo import GraspingAlgo, get_nearest_inst

np.set_printoptions(precision=7, suppress=True)

use_pump = True

# @benchmark
def infer(algo, img):
    return algo.infer_img(img)

def closeGripper(pump_ctrl):
    if pump_ctrl is not None:
        pump_ctrl.config_gripper(1)

def releaseGripper(pump_ctrl):
    if pump_ctrl is not None:
        pump_ctrl.config_gripper(0)

def validate_grasp_pose(pose):
    x, y, z = pose[0, 3], pose[1, 3], pose[2, 3]
    x_ok = abs(x) < 0.25 
    y_ok = y > -0.4 and y < -0.1
    z_ok = z > -0.2 and z < 0.55
    print(f'validating pos [{x:.4f}, {y:.4f}, {z:.4f}]', [{x_ok}, {y_ok}, {z_ok}])
    return x_ok and y_ok and z_ok

def click_callback(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, refPt_updated
        
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		refPt_updated = True


def world_pose_from_detected_obj(obj_pose_vec):
    obj_pose = detected_obj_pose_viewpoint 
    obj_pose[:3,3] = obj_pose_vec[:3]
    eef2obj = H_handeye @ obj_pose 
    H_base_to_obj = sensing_pose_eef @ H_handeye @ obj_pose
    return H_base_to_obj

def is_reachable(q_curr, target_pose_tcp):
    ec = 0 # NoError
    q_target = computeIK_kdl(q_curr, target_pose_tcp, ec)
    target_pose_fk = computeFK_kdl(q_target)
    print(f'isclose: {np.allclose(target_pose_fk, target_pose_tcp, atol=0.003)}')
    return np.allclose(target_pose_fk, target_pose_tcp, atol=0.003)

def move_if_reachable(grasp_pose_tcp):
    q_curr = eu_get_current_joint_positions()
    ec = 0 # NoError
    q_grasp = computeIK_kdl(q_curr, grasp_pose_tcp, ec)
    if np.max(np.abs(q_grasp)) > JLIMIT:
        print('\n===== joint out of range =====')
    elif not is_reachable(q_curr, grasp_pose_tcp):
        print('\n===== pose not reachable =====')
    else:
        eu_mov_to_target_jnt_pos(q_grasp)


def async_grasp(marching_dist):
    global pump_ctrl, GRASP_OFFSET
    print("===== stepping forward =====")  
    # moveRelativeAsync(PRE_GRASP_DIST + GRASP_OFFSET)
    # time.sleep(2.5)
    moveL_blk(marching_dist + GRASP_OFFSET)

    print("===== closing gripper =====")  
    closeGripper(pump_ctrl)
    time.sleep(1)

    print("===== rotating =====")  
    q_target = eu_get_current_joint_positions()
    q_target[5] += 90 / 180 * np.pi # 90deg
    eu_set_joint_velocities([2,2,3,5,2,10])
    
    eu_mov_to_target_jnt_pos(q_target)
    time.sleep(3)

    print("===== getting back =====")  
    moveL_blk(-0.1)

if __name__ == "__main__":  
    # EU Robot Arm initialization 
    if eu_init_device(1000):
        print("eu_init_device ok")
        if eu_get_heartbeat():
            print("eu_get_heartbeat ok")

            eu_enable_motor(True)
            eu_set_joint_max_velocity(10)
            eu_set_joint_velocities([2,2,3,5,2,10])
            eu_set_work_mode(ControlMode.POSITION_MODE)
            
            # move to initial pose            
            q_init = sensing_pose_list[sensing_pose_idx]
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

    # Vision Algo
    grasp_algo = GraspingAlgo()

    # Visualization
    cv2.namedWindow('test_grasping', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test_grasping', int(1280), int(720))
    cv2.setMouseCallback("test_grasping", click_callback)
    
    ## ================= Global Variables =================
    # EU arm control stuff
    j_idx = CLinkedList([1,2,3,4,5,6]).head
    q_curr = eu_get_current_joint_positions()
    JMOVE_STEP = 0.005
    q_target = q_init
    PRE_GRASP_DIST = 0.05 - 0.005 # mm
    GRIPPER_LENGTH = 0.135 # mm
    GRASP_OFFSET = 0.0275  
    saving_data_idx = 0
    refPt = (int(1280/2), int(720/2)) # center of img
    refPt_updated = False
    marching_dist = 0.
    JLIMIT = 3.1
    ## ================= Global Variables =================

    ## Data saving config ##
    date_prefix = datetime.date.today().isoformat()
    save_data_path = '/home/hkclr/AGBot_ws/eu_robot_ws/refactored_eu_arm/tmp/picking_data'
    grasping_data_p = os.path.join(save_data_path, "date_" + date_prefix)
    if not os.path.exists(grasping_data_p):
        os.makedirs(grasping_data_p)
    time_stamp = datetime.datetime.now().strftime("%H-%M-%S")
    grasping_data_p = os.path.join(grasping_data_p, time_stamp)
    if not os.path.exists(grasping_data_p):
        os.makedirs(grasping_data_p)

    try:
        while True:
            start = time.time()

            color_frame, depth_frame = rs_ctrl.get_stereo_frame()
            color_img = np.asanyarray(color_frame.get_data())

            # Visiualization and interactive control
            cv2.imshow('test_grasping', color_img)
            key = cv2.waitKey(1)

            if key & 0xFF == ord('q') or key == 27: # Press esc or 'q' to close the image window
                print("\n===== stop pipeline =====")  
                cv2.destroyAllWindows()
                break

            elif key & 0xFF == ord('s'):
                print("\n===== start inference =====")
                # Saving data
                pc = rs_ctrl.convert_to_pointcloud(color_frame, depth_frame)
                o3d.io.write_point_cloud(f'{grasping_data_p}/frame-{saving_data_idx:06d}.ply', pc)
                np.savetxt(f"{grasping_data_p}/frame-{saving_data_idx:06d}_pose.txt", q_curr, fmt="%.5f")
                cv2.imwrite(f'{grasping_data_p}/frame-{saving_data_idx:06d}_color.jpg', color_img)
                saving_data_idx += 1
                refPt_updated = False

                t0 = time.time()
                masks, bboxes = infer(grasp_algo, color_img)
                print(f'>>> infer {time.time()-t0}')

                if masks is not None and len(masks) == 0:
                    print('>'*15, 'no object detected', '>'*15)
                    continue
                    
                grasp_poses, grasp_uv = grasp_algo.gen_grasp_pose(color_img, pc, masks)
                img_vis = cv2.imread('/home/hkclr/AGBot_ws/eu_robot_ws/refactored_eu_arm/tmp/test_img_cups_.png')
                while not refPt_updated:
                    cv2.imshow('test_grasping', img_vis)
                    key = cv2.waitKey(500)
                    print('===== Click to select a grasp pose')

                if grasp_poses is None:
                    print('>'*15, 'Could not find valid grasp pose', '>'*15)
                    continue
                        
                grasp_idx = get_nearest_inst(grasp_uv, refPt)
                print(f'finding nearest inst from ref pt: {refPt}/[{grasp_uv[grasp_idx]}] [{grasp_idx}]->{grasp_poses[grasp_idx]}')

                q_curr = eu_get_current_joint_positions()
                sensing_pose_eef = computeFK_kdl(q_curr)       

                obj_pose_vec = grasp_poses[grasp_idx][0]
                # filter by robot workspace 
                obj_pose = detected_obj_pose_viewpoint 
                obj_pose[:3,3] = obj_pose_vec[:3]
                eef2obj = H_handeye @ obj_pose 
                H_base_to_obj = sensing_pose_eef @ H_handeye @ obj_pose

                print(f"H_base_to_obj: {H_base_to_obj}")

                if validate_grasp_pose(H_base_to_obj):
                    detected_obj_pose_viewpoint[:3,3] = obj_pose_vec[:3]
                    print(f'================detected pose: \n{repr(detected_obj_pose_viewpoint)}\n====================')  

                continue

            elif key & 0xFF == ord('g'):
                print("\n go grasping")  
                q_curr = eu_get_current_joint_positions()
                eef_pose = computeFK_kdl(q_curr)

                # pregrasp pose -> relative pose on x-y plane wrt eef
                eef2obj = H_handeye @ detected_obj_pose_viewpoint 
                eef2pre_grasp= np.eye(4)                
                eef2pre_grasp[:3, 3] = eef2obj[:3, 3]
                # handle z offset
                if eef2obj[2, 3] > PRE_GRASP_DIST + GRIPPER_LENGTH:
                    marching_dist = PRE_GRASP_DIST
                    eef2pre_grasp[2, 3] = eef2obj[2, 3] - GRIPPER_LENGTH - PRE_GRASP_DIST
                else:
                    marching_dist = eef2obj[2, 3] - GRIPPER_LENGTH
                    eef2pre_grasp[2, 3] = 0
                print(f'marching_dist: {eef2obj[2, 3] - PRE_GRASP_DIST - GRIPPER_LENGTH} | pre_grasp_dist: [{eef2pre_grasp[2, 3]}]')
                print(f'eef2pre_grasp: \n{repr(eef2pre_grasp)}')

                H_base_to_obj = eef_pose @ eef2pre_grasp
                grasp_pose_tcp = H_base_to_obj

                print(f'eef_pose: \n{repr(eef_pose)}')
                print(f'H_base_to_obj: \n{repr(H_base_to_obj)}')
                print(f'grasp_pose: \n{repr(grasp_pose_tcp)}')

                move_if_reachable(grasp_pose_tcp)
                continue

            elif key & 0xFF == ord('f'):
                print("\n Moving for Re-sensing & Locolization")  
                q_curr = eu_get_current_joint_positions()
                eef_pose = computeFK_kdl(q_curr)

                # pregrasp pose -> relative pose on x-y plane wrt eef
                eef2obj = H_handeye @ detected_obj_pose_viewpoint 
                eef2pre_grasp= np.eye(4)                
                eef2pre_grasp[:3, 3] = eef2obj[:3, 3]
                eef2pre_grasp[1, 3] -= 0.015
                # handle z offset
                if eef2obj[2, 3] > PRE_GRASP_DIST + GRIPPER_LENGTH:
                    eef2pre_grasp[2, 3] = eef2obj[2, 3] - GRIPPER_LENGTH - PRE_GRASP_DIST
                else:
                    eef2pre_grasp[2, 3] = 0

                H_base_to_obj = eef_pose @ eef2pre_grasp
                grasp_pose_tcp = H_base_to_obj
                move_if_reachable(grasp_pose_tcp)
                continue

            elif key & 0xFF == ord('t'):
                t_grasp = threading.Thread(target=async_grasp, args=(marching_dist,))
                t_grasp.start()
                continue

            elif key & 0xFF == ord('c'):
                sensing_pose_idx += 1
                idx = sensing_pose_idx % len(sensing_pose_list)
                sensing_pose = sensing_pose_list[idx]
                q_init = sensing_pose
                print(f'Moving to {idx+1}th sensing pose [{sensing_pose}]......\n')
                eu_mov_to_preset_jnt_pose(sensing_pose)

            elif key & 0xFF == ord('m'):
                print("\n moving along eef z axis")  
                moveRelativeAsync(0.01)
                continue

            elif key & 0xFF == ord('u'):
                print("\n moving along eef z axis")  
                moveRelativeAsync(-0.01)
                continue

            elif key & 0xFF == ord('v'):
                print("\n===== closing gripper =====")  
                closeGripper(pump_ctrl)
                continue

            elif key & 0xFF == ord('r'):
                print("\n===== releasing gripper =====")  
                releaseGripper(pump_ctrl)
                continue

            elif key & 0xFF == ord('i'):
                # q_init = np.array([-0.04506, -0.20009,  1.14981,  0.04487,  1.57482,  0.19491])
                eu_set_joint_velocities([2,2,3,5,2,10])
                eu_mov_to_target_jnt_pos(q_init)
                sensing_pose_idx = 0
                print(f'clicked xy: {refPt}')
                continue

            elif key & 0xFF == ord('p'):
                q_place = np.array([ 0.2346032, -0.8393921, -1.8604493,  0.077659 , -2.1227418, -1.5630305])
                eu_set_joint_velocities([2,2,3,5,2,2])
                eu_mov_to_preset_jnt_pose(q_place)
                continue

            elif key & 0xFF == ord('w'):
                q_target = eu_get_current_joint_positions()
                q_target[3] += JMOVE_STEP * 1.5
                eu_mov_to_target_jnt_pos(q_target)
                continue
            elif key & 0xFF == ord('e'):
                q_target = eu_get_current_joint_positions()
                q_target[3] -= JMOVE_STEP * 1.5
                eu_mov_to_target_jnt_pos(q_target)
                continue


            elif key & 0xFF == 82: # up 
                q_target = eu_get_current_joint_positions()
                q_target[j_idx.data-1] += JMOVE_STEP * 1.5
                eu_mov_to_target_jnt_pos(q_target)
                continue
            elif key & 0xFF == 84: # down
                q_target = eu_get_current_joint_positions()
                q_target[j_idx.data-1] -= JMOVE_STEP * 1.5
                eu_mov_to_target_jnt_pos(q_target)
                continue
            elif key & 0xFF == 81: # left 
                j_idx = j_idx.prev
                print(f"\n===== switch to prev joint, Joint [{j_idx.data}] selected =====")  
                continue
            elif key & 0xFF == 83: # right
                j_idx = j_idx.next
                print(f"\n===== switch to next joint, Joint [{j_idx.data}] selected =====")  
                continue
    finally:
        rs_ctrl.stop_streaming()
        grasp_algo.free_cuda_buffers()