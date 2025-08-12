import cv2
import numpy as np
import glob
from scipy.spatial.transform import Rotation as Rot
import copy, csv

from dev.rs_d405 import RealSenseController
from eu_arm.robot_arm_interface import RobotArm

from utils.circular_linked_list import CLinkedList


import sys
from pathlib import Path
project_path = Path(Path(__file__).parent.absolute()).parent.absolute()
sys.path.insert(0, str(project_path))
sys.path.insert(0, str(project_path)+'/src')

np.set_printoptions(precision=7, suppress=True)

import time


class MultiCameraRegistration(object):
    def __init__(self, rgb_intrinsic, relative_mat):
        self.project_idx_x = None
        self.project_idx_y = None
        self.rgb_intrinsic = rgb_intrinsic
        self.relative_mat = relative_mat
        self.eps = 1e-10

    def point_projection_vector(self, point):
        cam_fx = self.rgb_intrinsic[0, 0]
        cam_fy = self.rgb_intrinsic[1, 1]
        cam_cy = self.rgb_intrinsic[1, 2]
        cam_cx = self.rgb_intrinsic[0, 2]
        point_x = point[:, 0] / (point[:, 2] + self.eps)
        point_y = point[:, 1] / (point[:, 2] + self.eps)
        point_x = point_x * cam_fx + cam_cx
        point_y = point_y * cam_fy + cam_cy
        return point_x.astype(np.int32), point_y.astype(np.int32)

    def calculate_project_index(self, pc_array):
        pc_array = np.concatenate([pc_array, np.ones((pc_array.shape[0], 1))], axis=1)
        pc_rgb_array = np.linalg.inv(self.relative_mat).dot(pc_array.T).T[:, :3]
        self.project_idx_x, self.project_idx_y = self.point_projection_vector(pc_rgb_array)

    def get_rbgd_pointcloud(self, pc_array, rgb_img):
        self.calculate_project_index(pc_array)
        rgb_pc = rgb_img[np.clip(self.project_idx_y, 0, rgb_img.shape[0] - 1),
                         np.clip(self.project_idx_x, 0, rgb_img.shape[1] - 1), :]
        rgbd_pcd = o3d.geometry.PointCloud()
        rgbd_pcd.points = o3d.utility.Vector3dVector(pc_array)
        rgbd_pcd.colors = o3d.utility.Vector3dVector(rgb_pc[:, ::-1]/255.0)
        return rgbd_pcd

    def get_img_to_3d_mapping(self, pc_array, rgb_img):
        """
        map each (x,y) in rgb_img to real world x,y,z
        """
        self.calculate_project_index(pc_array)
        maxy, maxx = rgb_img.shape[:2]
        depth_img = np.zeros([maxy, maxx, 3], dtype=float)
        depth_img[:] = np.finfo(float).max
        for (point, rgbx, rgby) in zip(pc_array, self.project_idx_x, self.project_idx_y):
            if not ((0 <= rgbx < maxx) and (0 <= rgby < maxy)): continue
            if point[2] < depth_img[rgby, rgbx, 2]: depth_img[rgby, rgbx] = point
        return depth_img

    def imgXYs_to_3d_mapping(self, imgXYs, pc_array):
        """
        map (x,y) in rgb_img to real world x,y,z
        """
        self.calculate_project_index(pc_array)
        output = []
        for imgX, imgY in imgXYs:
            xy_diff = abs(imgX - self.project_idx_x) + abs(imgY - self.project_idx_y)
            output.append(pc_array[np.argmin(xy_diff)])
        return output

    def get_project_index(self):
        return self.project_idx_x, self.project_idx_y

def rigid_transform_3D(A, B):
    assert len(A) == len(B)

    N = A.shape[0]  # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    H = np.matmul(np.transpose(AA),BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.matmul(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
        print("Reflection detected")
        Vt[2, :] *= -1
        R = np.matmul(Vt.T,U.T)

    t = -np.matmul(R, centroid_A) + centroid_B
    # err = B - np.matmul(A,R.T) - t.reshape([1, 3])
    return R, t

def undistort(frame, k, d):
    h, w = frame.shape[:2]
    mapx, mapy = cv2.initUndistortRectifyMap(k, d, None, k, (w, h), 5)
    return cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)

# Define chessboard corners, return corner points
def make_chessboard_corners(w=11, h=8, corner_size_mm=0.006):
    # Construct chessboard corners matrix
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp*corner_size_mm  # mm
    return objp

def compute_reprojection_error(chess_corner3d, cam_corner3d, camera_other, trans_init):
    return 

use_real_robot = True
real_camera = True

if __name__ == "__main__":  
    robot = RobotArm(connect_robot=use_real_robot)
    q_init = [-0.0047937,  0.462778 ,  0.5702452,  0.3102525,-1.528397,  -0.985846 ]
    robot.moveJ(q_init)

    # Realsense init
    rs_ctrl = RealSenseController()
    if real_camera:
        rs_ctrl.config_stereo_stream()
    cam_intr, distor_coeff = rs_ctrl.get_intrinsics()
    # rs_ctrl.start_streaming()

    corners_pts = make_chessboard_corners()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001) # threshold 0.001

    # Visualization
    cv2.namedWindow('rs_calibration', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('rs_calibration', int(1920/2), int(1080/2))

    ## ================= Global Variables =================
    # EU arm control stuff
    j_idx = CLinkedList([1,2,3,4,5,6]).head
    q_curr, _ = robot.get_current_state()
    JMOVE_STEP = 0.005
    q_target = q_curr

    # Calibration stuff
    H_cam2obj = np.eye(4)
    H_base2eef = np.eye(4)
    H_cam2obj_list = []
    H_base2eef_list = []

    save_data_path = '/home/hkclr/AGBot_ws/eu_robot_ws/eu_robot/data'
    ## ================= Global Variables =================

    import time

    try:
        while True:
            start = time.time()

            # Find chessboard corners   
            color_img = rs_ctrl.get_color_frame()
            color_vis = color_img.copy()
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            ret_find_corners, corners = cv2.findChessboardCorners(gray, (11,8),None) # TODO: fix hardcode h,w
            if ret_find_corners == True:
                # refine corner to subpixel precision
                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(color_vis, (11,8), corners, ret_find_corners)

            # Visiualization and interactive control
            cv2.imshow('rs_calibration', color_vis)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27: # Press esc or 'q' to close the image window
                print("\n===== stop pipeline =====")  
                cv2.destroyAllWindows()
                break
            if key & 0xFF == ord('s'):
                print("\n===== solve calibration equation =====")  
                R_all_obj_to_cam=[]
                T_all_obj_to_cam=[] 
                R_all_base_to_eef=[] 
                T_all_base_to_eef = []

                # Compute handeye
                for i in range(len(H_cam2obj_list)):
                    R_all_obj_to_cam.append(H_cam2obj_list[i][:3,:3])
                    T_all_obj_to_cam.append(H_cam2obj_list[i][:3, 3].reshape((3,1)))

                    R_all_base_to_eef.append(H_base2eef_list[i][:3,:3])
                    T_all_base_to_eef.append(H_base2eef_list[i][:3, 3].reshape((3,1)))
                R, T = cv2.calibrateHandEye(R_all_base_to_eef, T_all_base_to_eef, R_all_obj_to_cam, T_all_obj_to_cam, method=cv2.CALIB_HAND_EYE_PARK)
                H_handeye = np.hstack((R, T))
                H_handeye = np.vstack((H_handeye, np.array([0, 0, 0, 1])))
                print(f'\n\n\n >>>>>>> CALIBRATION RESULT  \n{repr(H_handeye)} \n\n\n')

                # Compute reprojection error
                reproj_err_rvec = []
                reproj_err_tvec = []
                for i in range(len(H_cam2obj_list)):
                    H_cam2obj = H_cam2obj_list[i]
                    H_base2eef = H_base2eef_list[i]
                    H_base2obj = H_base2eef @ H_handeye @ H_cam2obj
                    reproj_err_rvec.append(Rot.from_matrix(H_base2obj[:3, :3]).as_rotvec())
                    reproj_err_tvec.append(H_base2obj[:3, 3])
                    print(f'rvec [{reproj_err_rvec[i].T}], tvec[{reproj_err_tvec[i].T}]')
                print("rotation vector variance:", np.var(reproj_err_rvec, axis=0), "translation variance:", np.var(reproj_err_tvec, axis=0))
                print("rvec diff:", np.max(reproj_err_rvec, axis=0) - np.min(reproj_err_rvec, axis=0), 
                      "tvec diff:", np.max(reproj_err_tvec, axis=0) - np.min(reproj_err_tvec, axis=0))
                break
            
            if key & 0xFF == ord('t'):
                print(f"\n===== take sample [{len(H_cam2obj_list)+1}] =====")  
                ret_pnp, rvec, tvec = cv2.solvePnP(corners_pts, corners, cam_intr, distor_coeff, flags=cv2.SOLVEPNP_IPPE) #, method=cv2.SOLVEPNP_IPPE
                if ret_pnp == True:
                    H_cam2obj = np.hstack(((cv2.Rodrigues(rvec))[0], tvec))
                    H_cam2obj = np.vstack((H_cam2obj, np.array([0, 0, 0, 1])))
                    print(f'rvec [{rvec.T}], tvec[{tvec.T}]')
                    print(f'Hmat pnp [{H_cam2obj}]')
                    H_cam2obj_list.append(H_cam2obj)

                # robot arm fk
                q_curr, eef_pose = robot.get_current_state()

                H_base2eef_list.append(eef_pose)
                print(f'eef_pose: {eef_pose}')

                # data logging & saving
                cv2.imwrite(f"{save_data_path}/calib_data/color_{len(H_cam2obj_list)}.jpg", color_img)
                np.savetxt( f"{save_data_path}/calib_data/H_cam2obj_{len(H_cam2obj_list)}.txt", H_cam2obj, fmt="%.5f")
                np.savetxt( f"{save_data_path}/calib_data/H_base2eef_{len(H_cam2obj_list)}.txt", eef_pose, fmt="%.5f")
                np.savetxt( f"{save_data_path}/calib_data/js_{len(H_cam2obj_list)}.txt", q_curr, fmt="%.7f")
                continue

            if key & 0xFF == 82: # up 
                q_target[j_idx.data-1] += JMOVE_STEP
                robot.moveJ(q_target)
                print(f'q_target: {q_target}')
                continue
            if key & 0xFF == 84: # down
                q_target[j_idx.data-1] -= JMOVE_STEP
                robot.moveJ(q_target)
                print(f'q_target: {q_target}')

                continue
            if key & 0xFF == 81: # left 
                print("\n===== switch to next joint =====")  
                j_idx = j_idx.prev
                print(f"\ Joint [{j_idx.data}] selected")
                continue
            if key & 0xFF == 83: # right
                print("\n===== switch to next joint =====")  
                j_idx = j_idx.next
                print(f"\ Joint [{j_idx.data}] selected")
                continue
    finally:
        rs_ctrl.stop_streaming()