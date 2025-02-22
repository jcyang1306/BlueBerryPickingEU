import os
import ctypes
import platform
import os.path
import numpy as np      

system = platform.system()
machine = platform.machine()

if system == "Linux":
    if machine == "x86_64":
        # os.environ['LD_LIBRARY_PATH'] = './lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        eu_lib = ctypes.CDLL("libeu_planet.so")
    elif machine.startswith("arm") or machine.startswith("aarch"):
        # os.environ['LD_LIBRARY_PATH'] = './lib:' + os.environ.get('LD_LIBRARY_PATH', '')
        eu_lib = ctypes.CDLL("libeu_planet.so")
elif system == "Windows":
    if machine == "AMD64" or machine == "x86_64":
        os.environ['PATH'] = './lib;' + os.environ.get('PATH', '')
        eu_lib = ctypes.CDLL("./lib/win_x64/eu_planet.dll")
else:
    raise OSError("Unsupported operating system.")

# Define constants
CAN_SUCCESS = 0

def print_planet_error_status(status):
    if status == 1:
        print("FAILED_ERRORDEVIECTYPE")
    elif status == 2:
        print("FAILED_DEVICEDISABLED")
    elif status == 3:
        print("FAILED_SETFAILED")
    elif status == 4:
        print("FAILED_MAXBYTESLIMIT")
    elif status == 5:
        print("FAILED_NORECEIVE")
    else:
        print("FAILED_UNKNOW")

# Define function prototypes
eu_lib.planet_initDLL.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
eu_lib.planet_initDLL.restype = ctypes.c_int

eu_lib.planet_freeDLL.argtypes = [ctypes.c_int]
eu_lib.planet_freeDLL.restype = ctypes.c_int

eu_lib.planet_getHeartbeat.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_bool), ctypes.c_int]
eu_lib.planet_getHeartbeat.restype = ctypes.c_int

eu_lib.planet_getSerialNumber.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
eu_lib.planet_getSerialNumber.restype = ctypes.c_int

eu_lib.planet_getHardwareVersion.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
eu_lib.planet_getHardwareVersion.restype = ctypes.c_int

eu_lib.planet_getFirmwareVersion.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_uint), ctypes.c_int]
eu_lib.planet_getFirmwareVersion.restype = ctypes.c_int

eu_lib.planet_getCurrent.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getCurrent.restype = ctypes.c_int

eu_lib.planet_getVelocity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getVelocity.restype = ctypes.c_int

eu_lib.planet_getPosition.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getPosition.restype = ctypes.c_int

eu_lib.planet_getTargetCurrent.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getTargetCurrent.restype = ctypes.c_int

eu_lib.planet_setTargetCurrent.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
eu_lib.planet_setTargetCurrent.restype = ctypes.c_int

eu_lib.planet_getTargetVelocity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getTargetVelocity.restype = ctypes.c_int

eu_lib.planet_setTargetVelocity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
eu_lib.planet_setTargetVelocity.restype = ctypes.c_int

eu_lib.planet_getTargetPosition.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getTargetPosition.restype = ctypes.c_int

eu_lib.planet_setTargetPosition.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
eu_lib.planet_setTargetPosition.restype = ctypes.c_int

eu_lib.planet_quick_setTargetPosition.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float]
eu_lib.planet_quick_setTargetPosition.restype = ctypes.c_int

eu_lib.planet_getTargetAcceleration.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getTargetAcceleration.restype = ctypes.c_int

eu_lib.planet_setTargetAcceleration.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
eu_lib.planet_setTargetAcceleration.restype = ctypes.c_int

eu_lib.planet_getTargetDeceleration.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getTargetDeceleration.restype = ctypes.c_int

eu_lib.planet_setTargetDeceleration.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
eu_lib.planet_setTargetDeceleration.restype = ctypes.c_int

eu_lib.planet_getMode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
eu_lib.planet_getMode.restype = ctypes.c_int

eu_lib.planet_setMode.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
eu_lib.planet_setMode.restype = ctypes.c_int

eu_lib.planet_getEnabled.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_bool), ctypes.c_int]
eu_lib.planet_getEnabled.restype = ctypes.c_int

eu_lib.planet_setEnabled.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int]
eu_lib.planet_setEnabled.restype = ctypes.c_int

eu_lib.planet_getStopRunState.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_bool), ctypes.c_int]
eu_lib.planet_getStopRunState.restype = ctypes.c_int

eu_lib.planet_setStopRunState.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_bool, ctypes.c_int]
eu_lib.planet_setStopRunState.restype = ctypes.c_int

eu_lib.planet_getAlert.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
eu_lib.planet_getAlert.restype = ctypes.c_int

eu_lib.planet_setMaxVelocity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.c_int]
eu_lib.planet_setMaxVelocity.restype = ctypes.c_int

eu_lib.planet_getMaxVelocity.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.c_int]
eu_lib.planet_getMaxVelocity.restype = ctypes.c_int

# Function implementations
def planet_init_dll(devType, devIndex, channel, baudrate):
    result = eu_lib.planet_initDLL(devType, devIndex, channel, baudrate)
    if result == CAN_SUCCESS:
        print("planet_initDLL successfully.")
        return True
    else:
        print("planet_initDLL failed.")
        print_planet_error_status(result)
        return False

def planet_free_dll(devIndex):
    result = eu_lib.planet_freeDLL(devIndex)
    if result == CAN_SUCCESS:
        print("planet_freeDLL successfully.")
        return True
    else:
        print("planet_freeDLL failed.")
        print_planet_error_status(result)
        return False

def planet_get_heartbeat(devIndex, id, timeOut=100):
    heartbeat = ctypes.c_bool()
    result = eu_lib.planet_getHeartbeat(devIndex, id, ctypes.byref(heartbeat), timeOut)
    if result == CAN_SUCCESS:
        return heartbeat.value
    else:
        print("Failed to get heartbeat.")
        print_planet_error_status(result)
        return None

def planet_get_serial_number(devIndex, id, timeOut=100):
    serialNum = ctypes.c_uint()
    result = eu_lib.planet_getSerialNumber(devIndex, id, ctypes.byref(serialNum), timeOut)
    if result == CAN_SUCCESS:
        return serialNum.value
    else:
        print("Failed to get serial number.")
        print_planet_error_status(result)
        return None

def planet_get_hardware_version(devIndex, id, timeOut=100):
    hwVersion = ctypes.c_uint()
    result = eu_lib.planet_getHardwareVersion(devIndex, id, ctypes.byref(hwVersion), timeOut)
    if result == CAN_SUCCESS:
        return hwVersion.value
    else:
        print("Failed to get hardware version.")
        print_planet_error_status(result)
        return None

def planet_get_firmware_version(devIndex, id, timeOut=100):
    fwVersion = ctypes.c_uint()
    result = eu_lib.planet_getFirmwareVersion(devIndex, id, ctypes.byref(fwVersion), timeOut)
    if result == CAN_SUCCESS:
        return fwVersion.value
    else:
        print("Failed to get firmware version.")
        print_planet_error_status(result)
        return None

def planet_get_current(devIndex, id, timeOut=100):
    current = ctypes.c_float()
    result = eu_lib.planet_getCurrent(devIndex, id, ctypes.byref(current), timeOut)
    if result == CAN_SUCCESS:
        return current.value
    else:
        print("Failed to get current.")
        print_planet_error_status(result)
        return None

#获得电机速度（rpm）
def planet_get_velocity(devIndex, id, timeOut=100):
    velocity = ctypes.c_float()
    result = eu_lib.planet_getVelocity(devIndex, id, ctypes.byref(velocity), timeOut)
    if result == CAN_SUCCESS:
        return velocity.value
    else:
        print("Failed to get velocity.")
        print_planet_error_status(result)
        return None
    
#获得电机位置（rad）
def planet_get_position(devIndex, id, timeOut=100):
    position = ctypes.c_float()
    result = eu_lib.planet_getPosition(devIndex, id, ctypes.byref(position), timeOut)
    if result == CAN_SUCCESS:
        return np.deg2rad(position.value)
    else:
        print("Failed to get position.")
        print_planet_error_status(result)
        return None
    
#获得目标电流值（q值）
def planet_get_target_current(devIndex, id, timeOut=100):
    targetCurrent = ctypes.c_float()
    result = eu_lib.planet_getTargetCurrent(devIndex, id, ctypes.byref(targetCurrent), timeOut)
    if result == CAN_SUCCESS:
        return targetCurrent.value
    else:
        print("Failed to get target current.")
        print_planet_error_status(result)
        return None
    
#设置目标电流值（q值）
def planet_set_target_current(devIndex, id, targetCurrent, timeOut=100):
    result = eu_lib.planet_setTargetCurrent(devIndex, id, ctypes.c_float(targetCurrent), timeOut)
    if result == CAN_SUCCESS:
        print("Target current set successfully.")
    else:
        print("Failed to set target current.")
        print_planet_error_status(result)

#获得目标速度值（rpm）
def planet_get_target_velocity(devIndex, id, timeOut=100):
    targetVelocity = ctypes.c_float()
    result = eu_lib.planet_getTargetVelocity(devIndex, id, ctypes.byref(targetVelocity), timeOut)
    if result == CAN_SUCCESS:
        return targetVelocity.value
    else:
        print("Failed to get target velocity.")
        print_planet_error_status(result)
        return None
    
#设置目标速度值（rpm）
def planet_set_target_velocity(devIndex, id, targetVelocity, timeOut=100):
    result = eu_lib.planet_setTargetVelocity(devIndex, id, ctypes.c_float(targetVelocity), timeOut)
    if result == CAN_SUCCESS:       
        # print("Target velocity set successfully.")
        return True
    else:
        print("Failed to set target velocity.")
        print_planet_error_status(result)
        return False

#获得目标位置值（rad）
def planet_get_target_position(devIndex, id, timeOut=100):
    targetPosition = ctypes.c_float()
    result = eu_lib.planet_getTargetPosition(devIndex, id, ctypes.byref(targetPosition), timeOut)
    if result == CAN_SUCCESS:
        # print(f'j{id}: val: {targetPosition.value}')
        return np.deg2rad(targetPosition.value)
    else:
        print("Failed to get target position.")
        print_planet_error_status(result)
        return None

#设置目标位置值（rad）
def planet_set_target_position(devIndex, id, targetPosition, timeOut=100):
    targetPosition = np.rad2deg(targetPosition)
    result = eu_lib.planet_setTargetPosition(devIndex, id, ctypes.c_float(targetPosition), timeOut)
    if result != CAN_SUCCESS:
        print("Failed to set target position.")
        print_planet_error_status(result)

#设置快速目标位置值（rad）
def planet_quick_set_target_position(devIndex, id, targetPosition):
    targetPosition = np.rad2deg(targetPosition)
    result = eu_lib.planet_quick_setTargetPosition(devIndex, id, ctypes.c_float(targetPosition))
    if result == CAN_SUCCESS:
        # print("Quick target position set successfully.")
        return True
    else:
        print("Failed to set quick target position.")
        print_planet_error_status(result)
        return False
    
#获得电机目标加速度（rpm/s）
def planet_get_target_acceleration(devIndex, id, timeOut=100):
    targetAcceleration = ctypes.c_float()
    result = eu_lib.planet_getTargetAcceleration(devIndex, id, ctypes.byref(targetAcceleration), timeOut)
    if result == CAN_SUCCESS:
        return targetAcceleration.value
    else:
        print("Failed to get target acceleration.")
        print_planet_error_status(result)
        return None
    
#设置目标加速度（rpm/s）
def planet_set_target_acceleration(devIndex, id, targetAcceleration, timeOut=100):
    result = eu_lib.planet_setTargetAcceleration(devIndex, id, ctypes.c_float(targetAcceleration), timeOut)
    if result == CAN_SUCCESS:
        print("Target acceleration set successfully.")
    else:
        print("Failed to set target acceleration.")
        print_planet_error_status(result)

#获得电机目标减速度（rpm/s）
def planet_get_target_deceleration(devIndex, id, timeOut=100):
    targetDeceleration = ctypes.c_float()
    result = eu_lib.planet_getTargetDeceleration(devIndex, id, ctypes.byref(targetDeceleration), timeOut)
    if result == CAN_SUCCESS:
        return targetDeceleration.value
    else:
        print("Failed to get target deceleration.")
        print_planet_error_status(result)
        return None
    
#设置目标减速度（rpm/s）
def planet_set_target_deceleration(devIndex, id, targetDeceleration, timeOut=100):
    result = eu_lib.planet_setTargetDeceleration(devIndex, id, ctypes.c_float(targetDeceleration), timeOut)
    if result == CAN_SUCCESS:
        print("Target deceleration set successfully.")
    else:
        print("Failed to set target deceleration.")
        print_planet_error_status(result)

def planet_get_mode(devIndex, id, timeOut=100):
    mode = ctypes.c_int()
    result = eu_lib.planet_getMode(devIndex, id, ctypes.byref(mode), timeOut)
    if result == CAN_SUCCESS:
        return mode.value
    else:
        print("Failed to get mode.")
        print_planet_error_status(result)
        return None

def planet_set_mode(devIndex, id, mode, timeOut=100):
    result = eu_lib.planet_setMode(devIndex, id, ctypes.c_int(mode), timeOut)
    if result == CAN_SUCCESS:
        print("Mode set successfully.")
    else:
        print("Failed to set mode.")
        print_planet_error_status(result)

def planet_get_enabled(devIndex, id, timeOut=100):
    enabled = ctypes.c_bool()
    result = eu_lib.planet_getEnabled(devIndex, id, ctypes.byref(enabled), timeOut)
    if result == CAN_SUCCESS:
        return enabled.value
    else:
        print("Failed to get enabled state.")
        print_planet_error_status(result)
        return None

def planet_set_enabled(devIndex, id, enabled, timeOut=100):
    result = eu_lib.planet_setEnabled(devIndex, id, ctypes.c_bool(enabled), timeOut)
    if result == CAN_SUCCESS:
        print("Enabled state set successfully.")
    else:
        print("Failed to set enabled state.")
        print_planet_error_status(result)

def planet_get_stop_run_state(devIndex, id, timeOut=100):
    stopRunState = ctypes.c_bool()
    result = eu_lib.planet_getStopRunState(devIndex, id, ctypes.byref(stopRunState), timeOut)
    if result == CAN_SUCCESS:
        return stopRunState.value
    else:
        print("Failed to get stop run state.")
        print_planet_error_status(result)
        return None

def planet_set_stop_run_state(devIndex, id, stopRunState, timeOut=100):
    result = eu_lib.planet_setStopRunState(devIndex, id, ctypes.c_bool(stopRunState), timeOut)
    if result == CAN_SUCCESS:
        print("Stop run state set successfully.")
    else:
        print("Failed to set stop run state.")
        print_planet_error_status(result)

def planet_get_alert(devIndex, id, timeOut=100):
    alert = ctypes.c_int()
    result = eu_lib.planet_getAlert(devIndex, id, ctypes.byref(alert), timeOut)
    if result == CAN_SUCCESS:
        return alert.value 
    else:
        print("Failed to get alert state.")
        print_planet_error_status(result)
        return None
    
#设置最大速度（rpm）
def planet_set_max_velocity(devIndex, id, maxVelocity, timeOut=100):
    result = eu_lib.planet_setMaxVelocity(devIndex, id, ctypes.c_float(maxVelocity), timeOut)
    if result == CAN_SUCCESS:
        print("Max velocity set successfully.")
    else:
        print("Failed to set max velocity.")
        print_planet_error_status(result)
        
#获得最大速度（rpm）
def planet_get_max_velocity(devIndex, id, timeOut=100):
    maxVelocity = ctypes.c_float()
    result = eu_lib.planet_getMaxVelocity(devIndex, id, ctypes.byref(maxVelocity), timeOut)
    if result == CAN_SUCCESS:
        return maxVelocity.value
    else:
        print("Failed to get max velocity.")
        print_planet_error_status(result)
