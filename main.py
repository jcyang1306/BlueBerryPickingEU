import numpy as np
import open3d as o3d
import cv2
import threading

from dev.rs_d405 import RealSenseController
from dev.pump_control import PumpControl
from eu_arm.robot_arm_interface import RobotArm
from utils.data_io import DataRecorder
# from env_cfg import *


if __name__ == '__main__':
    robot = RobotArm()
    q_curr, _ = robot.get_current_state()
    print(f'current joint pos: {q_curr}')

    # Realsense init
    rs_ctrl = RealSenseController()
    rs_ctrl.config_stereo_stream()

    color_frame, depth_frame = rs_ctrl.get_stereo_frame()
    color_img = np.asanyarray(color_frame.get_data())

    data_io = DataRecorder('tmp/release_output')

    data_io.save_img(color_img, 0)
