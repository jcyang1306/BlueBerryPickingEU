import numpy as np
import open3d as o3d
import cv2
import threading, time

from dev.rs_d405 import RealSenseController
from dev.pump_control import PumpControl
from eu_arm.robot_arm_interface import RobotArm
from utils.data_io import DataRecorder
from env_cfg import *

# TODO: resolve include path issue
from sensing_algo import GraspingAlgo


np.set_printoptions(precision=7, suppress=True)

def async_grasp(marching_dist):
    global pump_ctrl, GRASP_OFFSET
    print("===== stepping forward =====")  
    robot.marching(marching_dist + GRASP_OFFSET, [1,1,1,1,1,1])

    print("===== closing gripper =====")  
    closeGripper(pump_ctrl)
    time.sleep(1)

    print("===== rotating =====")  
    q_target = [0, 0, 0, 0, 0, 90 / 180 * np.pi]# 90deg for last joint
    robot.moveJ_relative(q_target, speed=[2,2,3,5,2,10])
    time.sleep(3)

    print("===== getting back =====")  
    robot.marching(-0.1, [1,1,1,1,1,1])

# Realman robot version grasp motion
def async_grasp_v2(marching_dist):
    global pump_ctrl, GRASP_OFFSET
    print("===== stepping forward =====")  
    fwd_motion = np.eye(4)
    fwd_motion[2, 3] = marching_dist + GRASP_OFFSET
    robot.moveL_relative_v2(fwd_motion, v=50)

    print("===== closing gripper =====")  
    closeGripper(pump_ctrl)
    time.sleep(1)

    print("===== rotating =====")  
    q_target_inc = [0, 0, 0, 0, 0, 90 / 180 * np.pi] # 90deg for last joint
    robot.moveJ_relative(q_target_inc, 100, 1)

    print("===== getting back =====")  
    bwd_motion = np.eye(4)
    bwd_motion[2, 3] = -0.1
    robot.moveL_relative_v2(bwd_motion, v=100)

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

def init_all(use_real_robot=True, use_pump=True, fake_camera=False):
    robot = RobotArm(connect_robot=use_real_robot)

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

    return robot, pump_ctrl, rs_ctrl, grasp_algo

class HiddenBlackboard():
    def __init__(self):
         pass

if __name__ == "__main__":  
    robot, pump_ctrl, rs_ctrl, grasp_algo = init_all()

    # move to initial pose            
    q_init = sensing_pose_list[sensing_pose_idx]
    robot.moveJ(q_init)

    # Visualization
    cv2.namedWindow('test_grasping', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test_grasping', int(1280), int(720))
    cv2.setMouseCallback("test_grasping", click_callback)

    ## ================= Global Variables =================
    # EU arm control stuff
    q_curr, _ = robot.get_current_state()
    j_idx = 0
    JMOVE_STEP = 0.005
    q_target = q_init
    PRE_GRASP_DIST = 0.05 - 0.005 # mm
    GRIPPER_LENGTH = 0.135 # mm
    GRASP_OFFSET = 0.015  
    saving_data_idx = 0
    refPt = (int(1280/2), int(720/2)) # center of img
    refPt_updated = False
    marching_dist = 0.
    JLIMIT = 3.1
    ## ================= Global Variables =================

    data_io = DataRecorder('tmp/picking_data')

    try:
        while True:
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
                pc = rs_ctrl.convert_to_pointcloud(color_frame, depth_frame)
                q_curr, sensing_pose_eef = robot.get_current_state()
                data_io.save_data(color_img, pc, q_curr)
                refPt_updated = False

                # Segmentation
                masks, bboxes = infer(grasp_algo, color_img)
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

                grasp_idx = GraspingAlgo.get_nearest_inst(grasp_uv, refPt)
                print(f'finding nearest inst from ref pt: {refPt}/[{grasp_uv[grasp_idx]}] [{grasp_idx}]->{grasp_poses[grasp_idx]}')    

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
                q_curr, eef_pose = robot.get_current_state()

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

                robot.move_to_pose(grasp_pose_tcp)
                continue

            elif key & 0xFF == ord('f'):
                print("\n Moving for Re-sensing & Locolization")  
                q_curr, eef_pose = robot.get_current_state()

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
                robot.move_to_pose(grasp_pose_tcp)
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
                robot.moveJ(sensing_pose, async_exec=True)

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
                robot.moveJ(q_init)
                sensing_pose_idx = 0
                print(f'clicked xy: {refPt}')
                continue

            elif key & 0xFF == ord('p'):
                q_place = np.array([ 0.2346032, -0.8393921, -1.8604493,  0.077659 , -2.1227418, -1.5630305])
                robot.moveJ(q_place)
                continue

            elif key & 0xFF == ord('w'):
                q_target, _ = robot.get_current_state()
                q_target[3] += JMOVE_STEP * 1.5
                robot.moveJ(q_target)
                continue
            elif key & 0xFF == ord('e'):
                q_target, _ = robot.get_current_state()
                q_target[3] -= JMOVE_STEP * 1.5
                robot.moveJ(q_target)
                continue

            elif key & 0xFF == 82: # up 
                q_target, _ = robot.get_current_state()
                q_target[j_idx] += JMOVE_STEP * 1.5
                robot.moveJ(q_target)
                continue
            elif key & 0xFF == 84: # down
                q_target, _ = robot.get_current_state()
                q_target[j_idx] -= JMOVE_STEP * 1.5
                robot.moveJ(q_target)
                continue
            elif key & 0xFF == 81: # left 
                j_idx = (j_idx - 1) % 6 # 6 DOF
                print(f"\n===== switch to prev joint, Joint [{j_idx+1}] selected =====")  
                continue
            elif key & 0xFF == 83: # right
                j_idx = (j_idx + 1) % 6 # 6 DOF
                print(f"\n===== switch to next joint, Joint [{j_idx+1}] selected =====")  
                continue

    finally:
        rs_ctrl.stop_streaming()
        grasp_algo.free_cuda_buffers()