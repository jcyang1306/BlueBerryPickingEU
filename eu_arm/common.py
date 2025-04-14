import numpy as np
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as Rot

#=======Gloabl variables for moveL control=======
JOINTLIMIT = 3.14
EEF_STEP = 0.0003

eef_step_fwd = np.eye(4)
eef_step_fwd[2, 3] = EEF_STEP

eef_step_bwd = np.eye(4)
eef_step_bwd[2, 3] = -EEF_STEP
#=======Gloabl variables for moveL control=======


def slerp(steps, rot):
    key_times = [0, 1] #rot0, rotf
    slerp = Slerp(key_times, rot)
    times = np.linspace(0, 1, steps)
    interp_rot = slerp(times)
    return interp_rot.as_matrix()

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