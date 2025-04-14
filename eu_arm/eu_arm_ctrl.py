from eu_arm.eu_planet import *
from eu_arm.eu_arm_const import *
import time

# 初始化设备
def eu_init_device(baudrate):
    devType = PLANET_DEV_TYPE_CANABLE
    channel = CHANNEL
    result = planet_init_dll(devType, EU_DEV_INDEX, channel, baudrate)
    if result:
        print("init_device successfully.")
        return True
    else:
        print("init_device failed.")
        return False

# 释放设备
def eu_free_device():
    result = planet_free_dll(EU_DEV_INDEX)
    if result:
        print("free_device successfully.")
    else:
        print("free_device failed.")
        
# 获取心跳
def eu_get_heartbeat():
    for id in range(1, JNT_NUM+1):
        result = planet_get_heartbeat(EU_DEV_INDEX, id)
        if result is None:  # 假设返回 0 表示成功，非 0 表示失败
            print(f"获取轴 {id} 的心跳失败")
            return False  # 返回失败
        # print(f"轴 {id} 的心跳为: {result}")
    return True  # 所有轴的心跳获取成功

# 上/下电
def eu_enable_motor(enable):
    for id in range(1, JNT_NUM+1):
        if enable:
            # 设置电机为使能状态
            planet_set_enabled(EU_DEV_INDEX, id, True, timeOut=100)
        else:
             # 设置电机为非使能状态
            planet_set_enabled(EU_DEV_INDEX, id, False, timeOut=100)

# 急停
def eu_stop_motor():
    for id in range(1, JNT_NUM+1):
        planet_set_stop_run_state(EU_DEV_INDEX, id, True, timeOut=100)

# 一键回零
# 参数:
#   velocity: 回零过程中的移动速度(rpm)
def eu_reset_position_zero(velocity):
    for id in range(1, JNT_NUM+1):
        planet_set_target_velocity(EU_DEV_INDEX, id, velocity, timeOut=100)
        planet_set_target_position(EU_DEV_INDEX, id, 0, timeOut=100)

# 设置机械臂各个关节的速度（各关节速度相同）
# 参数:
# velocity 是关节目标速度(rpm)
def eu_set_joint_velocity(velocity):
    for id in range(1, JNT_NUM+1):
        planet_set_target_velocity(EU_DEV_INDEX, id, velocity, timeOut=100)

# 设置机械臂各个关节的最大速度（各关节最大速度相同）
# 参数:
# velocity 是关节目标速度(rpm)
def eu_set_joint_max_velocity(velocity):
    for id in range(1, JNT_NUM+1):
        planet_set_max_velocity(EU_DEV_INDEX, id, velocity, timeOut=100)

# 设置机械臂各个关节的目标速度
# 参数:
# velocities 是一个包含六个关节目标速度的列表(rpm)
def eu_set_joint_velocities(velocities):
    for id in range(1, JNT_NUM+1):
        planet_set_target_velocity(EU_DEV_INDEX, id, velocities[id-1], timeOut=100)

# 获取当前机械臂各个关节的位置(rad)
def eu_get_current_joint_positions():
    positions = []
    for id in range(1, JNT_NUM+1):
        position = planet_get_position(EU_DEV_INDEX, id, timeOut=100)
        if position is None:
            raise Exception(f"Failed to get position for joint {id}")
        positions.append(position)
    return np.array(positions)

def eu_get_target_joint_positions():
    tgt_jpos = []
    for id in range(1, JNT_NUM+1):
        position = planet_get_target_position(EU_DEV_INDEX, id, timeOut=100)
        if position is None:
            raise Exception(f"Failed to get target position for joint {id}")
        tgt_jpos.append(position)
    return np.array(tgt_jpos)

# 移动至预置点位(传入目标点关节角)
# 参数:
# positions 是一个包含六个关节目标位置的列表(rad)
def eu_mov_to_preset_jnt_pose(positions):
    t0 = time.time()
    for id in range(JNT_NUM, 0, -1):
        planet_set_target_position(EU_DEV_INDEX, id, positions[id-1], timeOut=100)
    print(f'mov set target done, {time.time()-t0} used')

def eu_mov_to_target_jnt_pos(positions):
    for id in range(JNT_NUM, 0, -1):
        planet_quick_set_target_position(EU_DEV_INDEX, id, positions[id-1])

# 设置工作模式
def eu_set_work_mode(mode: ControlMode):
    for id in range(1, JNT_NUM + 1): 
        planet_set_mode(EU_DEV_INDEX,id, mode.value)

def check_all_close(target_jpos, jerr_thd_deg=0.2):
    curr_jpos = eu_get_current_joint_positions()
    diff_max = np.max(np.abs(target_jpos-curr_jpos))
    # print(f'diff_pos: {target_jpos-curr_jpos} -> {diff_max} [{diff_max<(jerr_thd*math.pi/180)}]')
    return diff_max<(jerr_thd_deg * np.pi / 180)

def moveJ_blk(target_jpos, jerr_thd=0.2):
    # TODO: check device init, running state... and handle exception
    eu_mov_to_target_jnt_pos(target_jpos)

    # checkAllClose at 200Hz freq
    while not check_all_close(target_jpos, jerr_thd):
        time.sleep(0.005)
    return True

# 调用示例
if __name__ == "__main__":  
    try:
        if not eu_init_device(BAUDRATE):
            raise RuntimeError("eu_init_device失败，终止操作")
        if not eu_get_heartbeat():
            raise RuntimeError("心跳获取失败，终止操作")
        eu_enable_motor(True)
        eu_set_joint_max_velocity(20)
        eu_reset_position_zero(10)
        eu_set_joint_velocity(20)
        positions = np.radians[0, 0, 90, 90, 0, 0]
        eu_mov_to_preset_jnt_pose(positions)        
        eu_free_device()
    except RuntimeError as e:
        print(e)
        exit()  # 终止程序