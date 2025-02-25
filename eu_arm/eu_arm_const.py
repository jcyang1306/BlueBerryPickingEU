from enum import Enum
import numpy as np

EU_DEV_INDEX = 2             # 设备索引
PLANET_DEV_TYPE_CANABLE = 11 # 意优canable设备
CHANNEL = 0
JNT_NUM = 6
THETA_LIMIT = np.rad2deg(175)
MAX_POS = np.rad2deg(150) 
BAUDRATE = 1000

# 定义控制模式枚举
class ControlMode(Enum):
    CONTOUR_POSITION_MODE = 1  # 轮廓位置控制模式
    VELOCITY_MODE = 3          # 轮廓速度模式
    CURRENT_MODE = 4           # 电流模式
    POSITION_MODE = 5          # 同步周期位置模式

# Calibrated DH Params New Arm KDL
kJNT_NUM = 6
kJOINT_TYPE = [0, 0, 0, 0, 0, 0]  # Revolute
kDH_A = np.array([0, 183.5816,164.8151, -0.1811, 1.7608, 0]) * 1e-3
kDH_ALPHA = np.array([90, 179.2158, 180.2342, -89.9757, 89.9942, 0])
kDH_D = np.array([109, 0.5024, -0.4520, 76.8241, 73.6426, 69.6]) * 1e-3
kDH_THETA = np.array([0.0133, 90.6272, 0.4330, -89.7143, 0.0299, -0.0002])