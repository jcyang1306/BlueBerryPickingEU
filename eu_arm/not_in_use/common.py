from eu_arm.eu_arm_interface import *
import threading
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as Rot
from utils.benchmark import benchmark

eu_arm = eu_arm_kinematics()

#=======Gloabl variables for moveL control=======
JOINTLIMIT = 3.14
EEF_STEP = 0.0003

eef_step_fwd = np.eye(4)
eef_step_fwd[2, 3] = EEF_STEP

eef_step_bwd = np.eye(4)
eef_step_bwd[2, 3] = -EEF_STEP
#=======Gloabl variables for moveL control=======


def check_all_close(target_jpos, jerr_thd_deg=0.2):
    curr_jpos = eu_get_current_joint_positions()
    diff_max = np.max(np.abs(target_jpos-curr_jpos))
    # print(f'diff_pos: {target_jpos-curr_jpos} -> {diff_max} [{diff_max<(jerr_thd*math.pi/180)}]')
    return diff_max<(jerr_thd_deg * np.pi / 180)


def computeFK_kdl(q):
    return eu_arm.computeFK(q)

def computeIK_kdl(q_init, pos_goal, retval):
    return eu_arm.computeIK(q_init, pos_goal, retval)

# @benchmark
def servoL(relative_pos):
    # robot arm fk
    q_curr = eu_get_current_joint_positions()
    eef_pose_frame = eu_arm.FK(q_curr)
    eef_pose = eu_arm.frame2mat(eef_pose_frame)
    
    grasp_pose = eef_pose @ relative_pos
    grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
    
    ec = 0 # NoError
    q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
    print(f'grasp js pose: {q_grasp}')
    print(f'delta q: {q_curr - q_grasp}')
    
    if np.max(np.max(q_grasp)) > JOINTLIMIT:
        print('\n===== joint out of range <exceed joint limit> =====')
    elif np.max(np.abs(q_curr - q_grasp) > 0.075):
        print('\n===== joint out of range <delta q> =====')
    else:
        eu_mov_to_target_jnt_pos(q_grasp)


def moveL(z_offset):
    eu_set_joint_velocities([1,1,1,1,1,1])
    stps_total = int(np.abs(z_offset)/EEF_STEP)
    if z_offset > 0:
        eef_step_mat = eef_step_fwd
    else:
        eef_step_mat = eef_step_bwd
    for i in range(stps_total):
        servoL(eef_step_mat)
        time.sleep(0.01)
    print(f"============moveL done [{stps_total}] ============")

def servoL_blk(relative_pos):
    # robot arm fk
    q_curr = eu_get_current_joint_positions()
    eef_pose_frame = eu_arm.FK(q_curr)
    eef_pose = eu_arm.frame2mat(eef_pose_frame)
    
    grasp_pose = eef_pose @ relative_pos
    grasp_pose_frame = eu_arm.mat2frame(grasp_pose)
    
    ec = 0 # NoError
    q_grasp = eu_arm.IK(q_curr, grasp_pose_frame, ec)
    print(f'grasp js pose: {q_grasp}')
    print(f'delta q: {q_curr - q_grasp}')
    
    if np.max(np.max(q_grasp)) > JOINTLIMIT:
        print('\n===== joint out of range <exceed joint limit> =====')
    elif np.max(np.abs(q_curr - q_grasp) > 0.075):
        print('\n===== joint out of range <delta q> =====')
    else:
        eu_mov_to_target_jnt_pos(q_grasp)


def moveJ_blk(target_jpos, jerr_thd=0.2):
    # TODO: check device init, running state... and handle exception
    eu_mov_to_target_jnt_pos(target_jpos)

    # checkAllClose at 200Hz freq
    while not check_all_close(target_jpos, jerr_thd):
        time.sleep(0.005)
    return True

def moveL_blk(z_offset):
    print(f'moving forward with [{z_offset}]')
    eu_set_joint_velocities([1,1,1,1,1,1])
    stps_total = int(np.abs(z_offset)/EEF_STEP)
    if z_offset > 0:
        eef_step_mat = eef_step_fwd
    else:
        eef_step_mat = eef_step_bwd

    # offline compute cartesian
    q0 = eu_get_current_joint_positions()
    eef_pose = eu_arm.computeFK(q0)
    q_curr = q0
    j_traj = []
    for i in range(stps_total):
        eef_pose = eef_pose @ eef_step_mat
        ec = 0
        traj_q = eu_arm.computeIK(q_curr, eef_pose, ec)
        q_curr = traj_q
        # print(f'step [{i}]: {q_curr}\n{eef_pose}')
        # if ec
        j_traj.append(traj_q)
    print(f'traj interpolation done, {len(j_traj)} pts in total')
    for q_target in j_traj:
        moveJ_blk(q_target, 0.5)

    # Waiting for final point execution cnverge
    while not check_all_close(j_traj[-1], 0.1):
        time.sleep(0.01)

    print(f"============moveL done [{stps_total}] ============")


def moveRelativeAsync(z_offset):
    t = threading.Thread(target=moveL_blk, args=(z_offset, ))
    t.start()


def slerp(steps, rot):
    key_times = [0, 1] #rot0, rotf
    slerp = Slerp(key_times, rot)
    times = np.linspace(0, 1, steps)
    interp_rot = slerp(times)
    return interp_rot.as_matrix()

@benchmark
def interpolate_pose(init_pose, target_pose, steps):
    # Rotation interpolation
    key_rots = Rot.from_matrix([init_pose[:3, :3],target_pose[:3, :3]])
    print(key_rots[0].as_matrix())
    interp_rots = slerp(steps, key_rots)

    # Translation interpolation
    tvec0 = init_pose[:3, 3]
    tvecf = target_pose[:3, 3]
    t = np.linspace(0, 1, steps).reshape(-1, 1)
    interp_transl = (tvec0 + t * (tvecf - tvec0)).reshape(-1, 3, 1)

    # Final pose array
    interp_T = np.concatenate([interp_rots, interp_transl], -1)
    tmp = np.tile(np.array([0, 0, 0, 1]), (steps, 1)).reshape(steps, 1, 4)
    interp_T = np.concatenate([interp_T, tmp], axis=1)

    # Perform IK and collision checking

    return interp_T