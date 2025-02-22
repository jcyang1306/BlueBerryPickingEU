import numpy as np
import time
from scipy.spatial.transform import Rotation as Rot
from copy import copy

from eu_arm_ctrl import *
from eu_kinematics import eu_arm_kinematics

# Planning libs
from roboticstoolbox import jtraj, ctraj
from ruckig import InputParameter, OutputParameter, Result, Ruckig

from spatialmath import *

from benchmark import benchmark

np.set_printoptions(precision=5, suppress=True)

def check_all_close(target_jpos, jerr_thd):
    curr_jpos = eu_get_current_joint_positions()
    # target_jpos = eu_get_target_joint_positions()
    # print(f'tgt_pos:  {target_jpos}')
    # print(f'curr_pos: {curr_jpos}')
    diff_max = np.max(np.abs(target_jpos-curr_jpos))
    # print(f'diff_pos: {target_jpos-curr_jpos} -> {diff_max} [{diff_max<(jerr_thd*math.pi/180)}]')
    return diff_max<(0.2*np.pi/180)

def jmove_blocking(target_jpos, jerr_thd=0.2):
    # TODO: check device init, running state... and handle exception
    print(f'>>> start moving to pos: {target_jpos}')
    eu_mov_to_preset_jnt_pose(target_jpos)

    # checkAllClose at 20Hz freq
    while not check_all_close(target_jpos, jerr_thd):
        time.sleep(0.01)
    print(f'>>> done moving to pos: {target_jpos}\n')

    # wait for motor to update state as it close to the target pos
    time.sleep(0.1)
    return True

# @benchmark
def Tmat2vec(T):
    tvec = T[:3,3].T
    r = Rot.from_matrix(T[:3,:3])
    rvec = r.as_euler('xyz')
    return np.hstack([tvec, rvec])

def vec2Tmat(affine_vec):
    tvec = np.array(affine_vec[:3])
    r = Rot.from_euler('xyz', affine_vec[3:], degrees=False)
    
    # assign to Tmat
    Tmat = np.eye(4)
    Tmat[:3,:3] = r.as_matrix()
    Tmat[:3,3] = tvec.T
    return Tmat


class eu_arm_interface():
    def __init__(self):
        self.init_and_set_default_behavior()
        self.kin = eu_arm_kinematics()
    def init_and_set_default_behavior(self):
        if eu_init_device(1000):
            print("eu_init_device ok")
            if eu_get_heartbeat():
                print("eu_get_heartbeat ok")

                eu_enable_motor(True)
                eu_set_joint_max_velocity(20)
                eu_set_joint_velocity(2)
                # eu_set_joint_velocities([3,4,5,5,5,7])

                eu_set_work_mode(ControlMode.POSITION_MODE)

    def computeIK(self, T, q0):
        return self.kin.MIK_from_T(T, q0)
    
    def computeFK(self, q):
        return self.kin.MFK(q)

    @benchmark
    def get_TCP_pose(self):
        q_curr = eu_get_current_joint_positions()
        return self.computeFK(q_curr)
    
    def get_j_pose(self):
        return eu_get_current_joint_positions()

    def move_relative(self, tvec):
        qlast = self.get_j_pose()
        T0 = self.get_TCP_pose()
        Tf = T0.copy()
        Tf[:3,3] += tvec

        dist_transl = np.linalg.norm(tvec)
        # print(f'dist_transl: {dist_transl}, delta t {dist_transl/75}')

        tg = ctraj(SE3(T0), SE3(Tf), 100)
        qtraj = [qlast]
        for Tsol in tg:
            qsol = self.computeIK(Tsol.A, qlast)
            qtraj.append(qsol)
            Te = Tsol.A - self.computeFK(qlast)
            qlast = qsol

            # print(f'Terr: {Te[:3,3]}')

        return qtraj

    def move_relative_ruckig(self, tvec):
        qlast = self.get_j_pose()
        T0 = self.get_TCP_pose()
        Tf = T0.copy()
        Tf[:3,3] += tvec

        dist_transl = np.linalg.norm(tvec)
        print(f'T0: {T0} \n Tf t {Tf}')

        s0 = Tmat2vec(T0)
        s1 = Tmat2vec(Tf)
        # ruckig
        # Create instances: the Ruckig OTG as well as input and output parameters
        kDoFs = 6
        otg = Ruckig(kDoFs, 0.03)  # DoFs, control cycle
        inp = InputParameter(kDoFs)
        out = OutputParameter(kDoFs)
    
        # Set input parameters
        inp.current_position = [0.3495981, -0.0365521,  0.2432336, -3.1324288, -0.0685687, 1.5569506]
        inp.current_velocity = np.zeros(kDoFs)
        inp.current_acceleration = np.zeros(kDoFs)
    
        inp.target_position = [0.2495981, -0.0965521,  0.2732336, -3.1324288, -0.0685687, 1.5569506]
        inp.target_velocity = np.zeros(kDoFs)
        inp.target_acceleration = np.zeros(kDoFs)
    
        inp.max_velocity =  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5
        inp.max_acceleration =  np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.4
        inp.max_jerk = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]) * 0.5
    
        print('\t'.join(['t'] + [str(i) for i in range(otg.degrees_of_freedom)]))
    
        # Generate the trajectory within the control loop
        first_output, out_list = None, []
        res = Result.Working
        while res == Result.Working:
            res = otg.update(inp, out)
    
            print('\t'.join([f'{out.time:0.3f}'] + [f'{p:0.3f}' for p in out.new_position]))
            out_list.append(copy(out))
    
            out.pass_to_input(inp)

            # solve ik
            new_state_vec = out.new_position
            Tep = vec2Tmat(new_state_vec)
            qsol = self.computeIK(Tep, qlast)
            qlast = qsol
            # print(f"Tep: {Tep}")
            print(f"qsol: {qsol}")
            eu_servoJ(qsol)
            time.sleep(0.03)

            if not first_output:
                first_output = copy(out)



        print(f's0: {repr(s0)}, s1: {repr(s1)}')

        return []

    def moveJ_native(self, qf):
        return jmove_blocking(qf)
    
def eu_servoJ(target_jpos):
    # for debugging usage
    q_curr = eu_get_current_joint_positions()
    q_target = eu_get_target_joint_positions()
    # print(f"q_diff: {q_target - q_curr}")

    return eu_mov_to_target_jnt_pos(target_jpos)

@benchmark
def plan_joint_space(q0, qf, t=100, qd0=0.0, qdf=0.0):
    tg = jtraj(q0, qf, t)
    # tg.plot(block=True)
    # for q in tg.q:
    #     print(q)
    return tg.q

def move_relative(T0, Tf, steps):
    tg = ctraj(T0, Tf, steps)


if __name__ == "__main__":  
    eu_arm = eu_arm_kinematics()
