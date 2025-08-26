# Third Party
import torch
import numpy as np
from typing import List, Optional, Sequence, Union
import time
from scipy.spatial.transform import Rotation as Rot

# CuRobo common
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml, get_robot_path
from curobo.geom.types import WorldConfig, Cuboid # not used? 
from curobo.util.logger import setup_curobo_logger #

# CuRobo robot model
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel

# Motion Gen
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.geom.sdf.world import CollisionCheckerType

# Rollout based planning (mpc & trajopt)
from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.mpc import MpcSolver, MpcSolverConfig
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig

tensor_args = TensorDeviceType()

class CuroboInterface:
    def __init__(self, robot_file="rml63.yml", world_file="collision_test.yml"):
        # Parse config files
        robot_cfg = load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"]
        world_cfg = {
            "cuboid": {
                "table": {
                    "dims": [0.1, 0.1, 0.1],  # x, y, z
                    "pose": [0.0, 0.0, -0.5, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
                },
            },
        }

        self.default_config = robot_cfg["kinematics"]["cspace"]["retract_config"]
        self.j_names = robot_cfg["kinematics"]["cspace"]["joint_names"]
        self.cached_state = None

        self.motion_gen = self.config_motion_gen(robot_cfg, world_cfg)
        self.ik_solver = self.config_ik_solve_single(robot_cfg, world_cfg)
        self.plan_config = MotionGenPlanConfig(
            enable_graph=False,
            enable_graph_attempt=4,
            max_attempts=10,
            time_dilation_factor=0.5,
            check_start_validity=False,
        )

    def FK(self, js_state, scalar_first=False) -> List[float]:
        """ scalar first order: (w, x, y ,z)
            scalar last  order: (x, y, z, w)
        """
        kin_state = self.motion_gen.kinematics.get_state(tensor_args.to_device(js_state))
        ee_pose_vec = np.ravel(kin_state.ee_pose.tolist()) # w, x, y, z

        # to xyzw order
        if scalar_first:
            ee_pose_vec[3:] = np.roll(ee_pose_vec[3:], 3)
        return ee_pose_vec.tolist()

    @staticmethod
    def config_motion_gen(robot_cfg="rml63.yml", world_cfg="collision_test.yml"):
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=True,
            interpolation_dt=0.03,
            collision_activation_distance=0.025,
            fixed_iters_trajopt=True,
            maximum_trajectory_dt=0.5,
            ik_opt_iters=500,
        )

        motion_gen = MotionGen(motion_gen_config)
        print("warming up...")
        t0 = time.time()
        motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        print(f"Curobo is Ready, [{time.time()-t0}]s used")

        return motion_gen

    @staticmethod
    def config_ik_solve_single(robot_cfg="rml63.yml", world_cfg="collision_test.yml"):
        ik_solve_cfg = IKSolverConfig.load_from_robot_config(
            robot_cfg,
            world_cfg,
            num_seeds = 100
        )
        ik_solver = IKSolver(ik_solve_cfg)
        return ik_solver

    def solveIK(self, pose_matrix):
        goal_pose = Pose.from_matrix(pose_matrix)
        result = self.ik_solver.solve_single(goal_pose)
        if not result.success:
            print(f'solveIK failed with overall error: [{result.error}] | \
                  t_err: [{result.position_error}] | r_err: [{result.rotation_error}]')
            return None
        # ik_solver.reset_seed() # TODO: figure out how this effect solve result 
        return result.js_solution.position.cpu().squeeze().tolist()

    def plan_single(self, start_state_js, goal_pose):
        curobo_goal_pose = Pose.from_list(goal_pose, q_xyzw=False)  
        curobo_start_state = JointState.from_position(
            tensor_args.to_device(start_state_js),
            joint_names=self.j_names,
        )

        result = self.motion_gen.plan_single(curobo_start_state.unsqueeze(0), curobo_goal_pose)

        # validate planning result
        success = result.success.item()
        error_msg = 'NO_ERROR'
        if not success:
            error_msg = result.status.name
            traj = None
        else:
            traj = result.interpolated_plan.position.cpu().numpy()

        print(f'Motion planning result: [{success}] <{error_msg}>')
        return traj


def test_rm_traj():
    sensing_pose = np.deg2rad([95.21700286865234, 57.68199920654297, -113.03099822998047, -9.659000396728516, -63.3650016784668, 176.83999633789062])
    back_pose    = np.deg2rad([109.25399780273438, 57.617000579833984, -145.7949981689453, -9.668999671936035, -48.00299835205078, 176.83999633789062])
    pre_grasp    = np.deg2rad([151.4980010986328, 38.560001373291016, -118.50199890136719, -9.645999908447266, -35.29899978637695, 176.83999633789062])
    grasp_pose   = np.deg2rad([151.32200622558594, 13.52400016784668, -114.81099700927734, -9.654000282287598, -12.645000457763672, 176.83799743652344])
    waypts_js = [sensing_pose, back_pose, pre_grasp, grasp_pose]

    cu_robot = CuroboInterface()

    ik_goal_pose = cu_robot.FK(pre_grasp) #wxyz
    traj1 = cu_robot.plan_single(sensing_pose, ik_goal_pose)
    final_waypt_js = traj1[-1].tolist()
    print(f'Valid final waypoint js: [{np.allclose(final_waypt_js, pre_grasp, atol=0.001)}]')
    import ipdb; ipdb.set_trace()

    ik_goal_pose = cu_robot.FK(grasp_pose) #wxyz
    traj2 = cu_robot.plan_single(pre_grasp, ik_goal_pose)
    final_waypt_js = traj2[-1].tolist()
    print(f'Valid final waypoint js: [{np.allclose(final_waypt_js, grasp_pose, atol=0.001)}]')
    import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    test_rm_traj()