from eu_arm.eu_arm_ctrl import *
from eu_arm.eu_kinematics import eu_arm_kinematics
from eu_arm.common import *

class RobotArm():
    def __init__(self, connect_robot=True):
        if connect_robot == True:
            self.try_connect_robot()
        self.eu_kin_ = eu_arm_kinematics()

    def try_connect_robot(self):
        try:
            eu_init_device(baudrate=1000)
            eu_enable_motor(True)
            eu_set_joint_max_velocity(10)
            eu_set_joint_velocities([2,2,3,5,2,10])
            eu_set_work_mode(ControlMode.POSITION_MODE)
        except RuntimeError as e:
            print(f'Init robot arm failed with error: [{e}]')
            exit()
        print(f'Init robot arm done')       
    
    def get_current_state(self):
        curr_q = eu_get_current_joint_positions()
        curr_pose = self.eu_kin_.computeFK(curr_q)
        return curr_q, curr_pose
    
    def FK_KDL(self, q):
        return self.eu_kin_.computeFK(q)
    
    def IK_KDL(self, q_init, pos_goal, retval):
        return self.eu_kin_.computeIK(q_init, pos_goal, retval)
    
    def moveJ(self, target_jpos, jerr_thd=0.2, async_exec=True):
        if async_exec:
            return eu_mov_to_target_jnt_pos(target_jpos, jerr_thd)
        else:
            return moveJ_blk(target_jpos, jerr_thd)
        
    def marching(self, marching_dist, speed=[1,1,1,1,1,1]):
        print(f'moving forward with [{marching_dist}]')
        eu_set_joint_velocities(speed)
        steps_total = int(np.abs(marching_dist)/EEF_STEP)
        if marching_dist > 0:
            eef_step_mat = eef_step_fwd
        else:
            eef_step_mat = eef_step_bwd

        # offline compute cartesian
        q0 = eu_get_current_joint_positions()
        eef_pose = self.eu_kin_.computeFK(q0)
        q_curr = q0
        j_traj = []
        for i in range(steps_total):
            eef_pose = eef_pose @ eef_step_mat
            traj_q = self.eu_kin_.computeIK(q_curr, eef_pose, retval=0)
            q_curr = traj_q
            j_traj.append(traj_q)
        print(f'traj interpolation done, {len(j_traj)} pts in total')
        for q_target in j_traj:
            moveJ_blk(q_target, jerr_thd=0.5)

        # Waiting for final point execution cnverge
        while not check_all_close(j_traj[-1], 0.1):
            time.sleep(0.01)

    def move_to_pose(self, target_pose_tcp):
        # Check reachability
        q_curr, _ = self.get_current_state()
        retval = 0
        q_target = self.IK_KDL(q_curr, target_pose_tcp, retval)
        target_pose_fk = self.FK_KDL(q_target)

        if np.max(np.abs(q_target)) > 3.1: # TODO: fix hardcode
            print('\n===== joint out of range =====')
        elif np.allclose(target_pose_fk, target_pose_tcp, atol=0.003):
            print(f'\n===== pose not reachable [{retval}]=====')
        else:
            return self.moveJ(q_target)

    def moveL(self, pose, speed, async_exec=False):
        t0 = time.time()
        q_init, pose_init = self.get_current_state()
        pose_final = pose_init @ pose
        print(pose_final)

        q_final = self.IK_KDL(q_init, pose_final, 0)
        q_diff = q_final - q_init
        JNT_MAX_VEL = np.array([1,1,1,1,1,1]) # TODO: move to config and set a more reasonable value
        # steps = int(np.max(np.abs(q_diff) / JNT_MAX_VEL)) # rough computation of steps
        steps = 100
        print(f'MoveL plan with steps: {steps}')

        # Interpolation, solveIK and compose traj
        interp_poses = interpolate_pose(pose_init, pose_final, steps)
        q_traj = []
        q_last = q_init
        for pose in interp_poses:
            q_waypt = self.IK_KDL(q_last, pose, 0)
            q_last = q_waypt
            q_traj.append(q_waypt)

        # Traj execution
        for q_waypt in q_traj:
            if async_exec:
                # moveJ_blk(q_waypt, jerr_thd=0.5)
                pass
            else:
                moveJ_blk(q_waypt, jerr_thd=0.5)
        print(f'time elapsed: {time.time() - t0}')

    def moveJ(self, q_target, async_exec=True):
        return moveJ_blk(q_target)


if __name__ == '__main__':
    robot = RobotArm()
    tcp_offset = np.eye(4)
    tcp_offset[:3, 3] = np.array([0,0,0.1])

    q_init = [-0.07507,  0.20306,  0.47601,  0.47803, -1.48461,  1.65421]
    robot.moveJ(q_init)
    q, tcp_pose = robot.get_current_state()
    print(f'[Init] current state q: {q}\n{tcp_pose}')


    robot.moveL(tcp_offset, 0.1)
    q, tcp_pose = robot.get_current_state()
    print(f'[Final] current state q: {q}\n{tcp_pose}')

    rot_left = np.array([
        [1.0000000,  0.0000000,  0.0000000,  0.0000000],
        [0.0000000,  0.8660254, -0.5000000,  0.0000000],
        [0.0000000,  0.5000000,  0.8660254,  0.0000000],
        [0.0000000,  0.0000000,  0.0000000,  1.0000000]
    ])
    robot.moveL(rot_left, 0.1)

    rot_up = np.array([
        [0.8660254,  0.0000000,  0.5000000, 0.0000000],
        [0.0000000,  1.0000000,  0.0000000, 0.0000000],
        [-0.5000000, 0.0000000,  0.8660254, 0.0000000],
        [0.0000000,  0.0000000,  0.0000000,  1.0000000]
    ])
    rot_left2up = np.linalg.inv(rot_left) @ rot_up
    robot.moveL(rot_left2up, 0.1)

    rot_right = np.array([
        [1.0000000,  0.0000000,  0.0000000,  0.0000000],
        [0.0000000,  0.8660254,  0.5000000,  0.0000000],
        [0.0000000, -0.5000000,  0.8660254,  0.0000000],
        [0.0000000,  0.0000000,  0.0000000,  1.0000000]
    ])
    rot_up2right = np.linalg.inv(rot_up) @ rot_right
    robot.moveL(rot_up2right, 0.1)

    rot_down = np.array([
        [0.8660254,  0.0000000, -0.5000000, 0.0000000],
        [0.0000000,  1.0000000,  0.0000000, 0.0000000],
        [0.5000000,  0.0000000,  0.8660254, 0.0000000],
        [0.0000000,  0.0000000,  0.0000000,  1.0000000]
    ])
    rot_right2down = np.linalg.inv(rot_right) @ rot_down
    robot.moveL(rot_right2down, 0.1)


    rot_down2init = np.linalg.inv(rot_down)
    robot.moveL(rot_down2init, 0.1)