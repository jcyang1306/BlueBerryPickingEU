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

# Calibrated DH Params
kJOINT_TYPE = [0, 0, 0, 0, 0, 0]  # Revolute
kDH_A = np.array([0, 0, 183.765, 164.6548, -0.4715, -1.7394]) * 1e-3
kDH_ALPHA = np.array([0, 90, 179.3582, 179.6965, -90.2676, 90.0063]) * np.pi / 180.
kDH_D = np.array([109, 0.7599, -0.1203, 76.2699, 73.7185, 69.6]) * 1e-3
kDH_THETA = np.array([-0.2988, 88.8208, 0.0709, -90.3032, -0.0696, 0.0002]) * np.pi / 180.