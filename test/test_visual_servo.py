import numpy as np
import open3d as o3d
import cv2
import threading, time
import sys, os
import math

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dev.rs_d405 import RealSenseController
from rm_arm.core.rm_robot_controller import RobotArmController
from utils.data_io import DataRecorder
from env_cfg import *

class PIDImpl:
    def __init__(self, dt=0.01, max_val=0.03, min_val=-0.03, Kp=0.8, Kd=0.1, Ki=0.05):
        self.dt = dt
        self.max = max_val
        self.min = min_val
        self.Kp = Kp
        self.Kd = Kd
        self.Ki = Ki
        self.alpha = 0.5
        self.prev_err = 0
        self.integral = 0

    def calculate(self, err):
        ef = self.alpha * err + (1 - self.alpha) * self.prev_err

        # Integral term
        self.integral += err * self.dt
        
        # Integral windup avoidance
        self.integral = min(self.integral, 0.05)
        self.integral = max(self.integral, -0.05)  # Fixed from original C++ code which had a bug
        
        if (err > 0) != (self.integral > 0):
            self.integral = 0

        # Derivative term
        diff = (err - self.prev_err) / self.dt

        # Calculate total output
        output = self.Kp * ef + self.Kd * diff
        if -0.005 < err < 0.005:
            output += self.Ki * self.integral

        # Restrict to max/min
        output = min(output, self.max)
        output = max(output, self.min)

        # Save err to previous err
        self.prev_err = err
        return output

class PosePIDController():
    def __init__(self):
        DELTA_ROT = 0.05
        self.controllers_ = [
            PIDImpl(0.01, 0.004, -0.004), # tx
            PIDImpl(0.01, 0.004, -0.004), # ty
            PIDImpl(0.01, 0.0075, -0.0075), # tz
            PIDImpl(0.01, DELTA_ROT, -DELTA_ROT), # rx
            PIDImpl(0.01, DELTA_ROT, -DELTA_ROT), # ry
            PIDImpl(0.01, DELTA_ROT, -DELTA_ROT)  # rz
        ]
        
    def calculate(self, dst_pose_vec: list[float], curr_pose_vec: list[float]):
        err_pose_vec =  [dst - src for dst, src in zip(dst_pose_vec, curr_pose_vec)]
        delta_pose_vec = [
            pid.calculate(err) for pid, err in zip(self.controllers_, err_pose_vec)
        ]
        return delta_pose_vec
        
# Define chessboard corners, return corner points
def make_chessboard_corners(w=11, h=8, corner_size_mm=0.006):
    # Construct chessboard corners matrix
    objp = np.zeros((w*h,3), np.float32)
    objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1,2)
    objp = objp*corner_size_mm  # mm
    return objp

corners_pts = make_chessboard_corners()
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 150, 0.001) # threshold 0.001
_lck = threading.Lock()
_pid = PosePIDController()
_stop_ctrl = False
_ctrl_step = False

T_base2eef = np.eye(4)
T_desired_cam2obj = np.array([   
    [ 1        ,  0        ,  0        , -0.043154],
    [ 0        ,  1        ,  0        , -0.0512194],
    [ 0        ,  0        ,  1        ,  0.2019793],
    [ 0.       ,  0.       ,  0.       ,  1.       ]
])

T_dobj2cam = np.linalg.inv(T_desired_cam2obj)
def rm_control_loop(robot):
    global _ctrl_step, _stop_ctrl, _lck, _pid, \
            T_base2eef, T_base2tgteef
    
    while not _stop_ctrl:
        _, robot_pose = robot.get_current_state(fmt='matrix')
        pose_vec = robot.pose_mat2vec(robot_pose)
        
        #============mutex lock============
        _lck.acquire()
        
        # update robot state
        T_base2eef = robot_pose
        dst_pose_vec = robot.pose_mat2vec(T_base2tgteef) # xyzrpy
        
        _lck.release()
        #============mutex lock============
        
        # compute vc and perform control
        if _ctrl_step:
            delta_pose_vec = _pid.calculate(dst_pose_vec, pose_vec)
            next_pose_vec = [val1 + val2 for val1, val2 in zip(pose_vec, delta_pose_vec)]
            # _ctrl_step = False
            print(f'\ncurr_pose {pose_vec}')
            print(f'dst_pose_vec {dst_pose_vec}')
            print(f'delta_pose {delta_pose_vec}')
            print(f'next_pose_vec {next_pose_vec}')
            
        else:
            next_pose_vec = pose_vec
            print(f'target pose unchanged {next_pose_vec}')
        # import ipdb; ipdb.set_trace()
        robot.servoL(next_pose_vec)
        time.sleep(0.02) # 50 Hz control freq
    
np.set_printoptions(precision=7, suppress=True)
if __name__ == "__main__":  
    tf_handeye = np.array([
      [ 0.00097161, 0.99990134,-0.01401297,-0.05197567],
      [-0.9999314 , 0.00080787,-0.01168531, 0.00909704],
      [-0.01167283, 0.01402336, 0.99983353, 0.00952641],
      [ 0         , 0         , 0         , 1         ]

    ])
    print(f'H_handeye: {tf_handeye}')

    # Robot init
    robot = RobotArmController("192.168.1.18", 8080, 3)
    ret = robot.robot.rm_change_work_frame("Base")
    robot.get_arm_software_info()
    
    # global vars
    _, T_base2eef = robot.get_current_state(fmt='matrix')
    T_base2tgteef = T_base2eef
    
    # Realsense init
    rs_ctrl = RealSenseController()
    rs_ctrl.config_stereo_stream()
    cam_intr, distor_coeff = rs_ctrl.get_intrinsics()

    # move to initial pose            
    q_init = [0.07353071825730682, 0.4344473608685379, -1.949602629193059, -0.0020245818740567733, -1.6133998468592758, 3.774518985625807]
    robot.moveJ(q_init)

    # Visualization
    cv2.namedWindow('test_PBVS', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('test_PBVS', int(1280), int(720))

    t_rm_ctrl = threading.Thread(target=rm_control_loop, args=(robot, ))
    t_rm_ctrl.start()

    try:
        while True:
            start = time.time()
            color_img = rs_ctrl.get_color_frame()
            color_vis = color_img.copy()
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
            
            t0 = time.time()
            # ret_find_corners, corners = cv2.findChessboardCorners(gray, (11,8), cv2.CALIB_CB_FAST_CHECK)
            ret_find_corners, corners = cv2.findChessboardCorners(gray, (11,8))
            
            if ret_find_corners == True:
                # refine corner to subpixel precision
                cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
                cv2.drawChessboardCorners(color_vis, (11,8), corners, ret_find_corners)

                ret_pnp, rvec, tvec = cv2.solvePnP(corners_pts, corners, cam_intr, distor_coeff, flags=cv2.SOLVEPNP_IPPE) #, method=cv2.SOLVEPNP_IPPE
                if ret_pnp == True:
                    T_cam2obj = np.hstack(((cv2.Rodrigues(rvec))[0], tvec))
                    T_cam2obj = np.vstack((T_cam2obj, np.array([0, 0, 0, 1])))
                # print(f'T_cam2obj: {repr(T_cam2obj)}')
                
            #============mutex lock============
            _lck.acquire()
            # import ipdb; ipdb.set_trace()
            
            T_base2obj = T_base2eef @ tf_handeye @ T_cam2obj
            T_base2tgteef = T_base2obj @ T_dobj2cam @ np.linalg.inv(tf_handeye)
            print(f'\ncurr pose:\n{T_base2eef}')
            print(f'target pose:\n{T_base2tgteef}')
            
            _lck.release()
            #============mutex lock============
            
            # print(f'detect corners time: {time.time()-t0:.3f}s')

            # Visiualization and interactive control
            cv2.imshow('test_PBVS', color_vis)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27: # Press esc or 'q' to close the image window
                print("\n===== stop pipeline =====")  
                cv2.destroyAllWindows()
                break
            if key & 0xFF == ord('s'):
                print("\n===== control step =====")
                _ctrl_step = not _ctrl_step
                continue
    finally:
        rs_ctrl.stop_streaming()
        _stop_ctrl = True