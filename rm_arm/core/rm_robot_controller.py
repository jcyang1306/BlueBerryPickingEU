import sys
import os
import numpy as np
from scipy.spatial.transform import Rotation as Rot

# Add the parent directory of src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Robotic_Arm.rm_robot_interface import *

# 定义机械臂型号到点位的映射  
arm_models_to_points = {  
    "RM_65": [  
        [0, 0, 0, 0, 0, 0],
        [-0.3, 0, 0.3, 3.14, 0, 0],
        [
            [-0.3, 0, 0.3, 3.14, 0, 0],
            [-0.27, -0.22, 0.3, 3.14, 0, 0],
            [-0.314, -0.25, 0.2, 3.14, 0, 0],
            [-0.239, 0.166, 0.276, 3.14, 0, 0],
            [-0.239, 0.264, 0.126, 3.14, 0, 0]
        ]  
    ],  
    "RM_75": [  
        [0, 20, 0, 70, 0, 90, 0],    
        [0.297557, 0, 0.337061, 3.142, 0, 3.142],    
        [
            [0.3, 0.1, 0.337061, 3.142, 0, 3.142],
            [0.2, 0.3, 0.237061, 3.142, 0, 3.142],
            [0.2, 0.25, 0.037061, 3.142, 0, 3.142],
            [0.1, 0.3, 0.137061, 3.142, 0, 3.142],
            [0.2, 0.25, 0.337061, 3.142, 0, 3.142]
        ]
    ], 
    "RML_63": [  
        [0, 20, 70, 0, 90, 0],
        [0.448968, 0, 0.345083, 3.142, 0, 3.142],
        [
            [0.3, 0.3, 0.345083, 3.142, 0, 3.142],
            [0.3, 0.4, 0.145083, 3.142, 0, 3.142],
            [0.3, 0.2, 0.045083, 3.142, 0, 3.142],
            [0.4, 0.1, 0.145083, 3.142, 0, 3.142],
            [0.5, 0, 0.345083, 3.142, 0, 3.142]
        ]  
    ], 
    "ECO_65": [  
        [0, 20, 70, 0, -90, 0],
        [0.352925, -0.058880, 0.327320, 3.141, 0, -1.57],
        [
            [0.3, 0.3, 0.327320, 3.141, 0, -1.57],
            [0.2, 0.4, 0.127320, 3.141, 0, -1.57],
            [0.2, 0.2, 0.027320, 3.141, 0, -1.57],
            [0.3, 0.1, 0.227320, 3.141, 0, -1.57],
            [0.4, 0, 0.327320, 3.141, 0, -1.57]
        ]
    ],
    "GEN_72": [  
        [0, 0, 0, -90, 0, 0, 0],
        [0.359500, 0, 0.426500, 3.142, 0, 0],
        [
            [0.359500, 0, 0.426500, 3.142, 0, 0],
            [0.2, 0.3, 0.426500, 3.142, 0, 0],
            [0.2, 0.3, 0.3, 3.142, 0, 0],
            [0.3, 0.3, 0.3, 3.142, 0, 0],
            [0.3, -0.1, 0.4, 3.142, 0, 0]
        ] 
    ],
    "ECO_63": [  
        [0, 20, 70, 0, -90, 0],
        [0.544228, -0.058900, 0.468274, 3.142, 0, -1.571],
        [
            [0.3, 0.3, 0.468274, 3.142, 0, -1.571],
            [0.3, 0.4, 0.168274, 3.142, 0, -1.571],
            [0.3, 0.2, 0.268274, 3.142, 0, -1.571],
            [0.4, 0.1, 0.368274, 3.142, 0, -1.571],
            [0.5, 0, 0.468274, 3.142, 0, -1.571]
        ]  
    ],
}

class RobotArmController:
    def __init__(self, ip, port, level=3, mode=2):
        self.thread_mode = rm_thread_mode_e(mode)
        self.robot = RoboticArm(self.thread_mode)
        self.handle = self.robot.rm_create_robot_arm(ip, port, level)

        if self.handle.id == -1:
            print("\nFailed to connect to the robot arm\n")
            exit(1)
        else:
            print(f"\nSuccessfully connected to the robot arm: {self.handle.id}\n")

    def disconnect(self):
        handle = self.robot.rm_delete_robot_arm()
        if handle == 0:
            print("\nSuccessfully disconnected from the robot arm\n")
        else:
            print("\nFailed to disconnect from the robot arm\n")

    def get_arm_model(self):
        res, model = self.robot.rm_get_robot_info()
        if res == 0:
            return model["arm_model"]
        else:
            print("\nFailed to get robot arm model\n")

    def get_arm_software_info(self):
        software_info = self.robot.rm_get_arm_software_info()
        if software_info[0] == 0:
            print("\n================== Arm Software Information ==================")
            print("Arm Model: ", software_info[1]['product_version'])
            print("Algorithm Library Version: ", software_info[1]['algorithm_info']['version'])
            print("Control Layer Software Version: ", software_info[1]['ctrl_info']['version'])
            print("Dynamics Version: ", software_info[1]['dynamic_info']['model_version'])
            print("Planning Layer Software Version: ", software_info[1]['plan_info']['version'])
            print("==============================================================\n")
        else:
            print("\nFailed to get arm software information, Error code: ", software_info[0], "\n")

    @staticmethod
    def pose_mat2vec(pose_mat):
        rpy = Rot.from_matrix(pose_mat[0:3,0:3]).as_euler('xyz', degrees=False)
        position = pose_mat[0:3, 3]
        
        return  [position[0], position[1], position[2], \
                    rpy[0], rpy[1], rpy[2]]    


    def get_joint_state(self, degrees=False):
        ret, curr_jq = self.robot.rm_get_joint_degree()
        if ret == 0:
            if not degrees:
                curr_jq = [jq / 180 * np.pi for jq in curr_jq]
            return curr_jq
        else:
            print("\nget_current_joint_angles motion failed, Error code: ", ret, "\n")
            return [] 

    def get_current_state(self, fmt: str='xyzrpy'):
        ret, state = self.robot.rm_get_current_arm_state()
        if ret != 0:
            print("Get current state failed, Error code: ", ret)
            return None

        curr_q = [jq / 180 * np.pi for jq in state['joint']]
        pose_xyzrpy = state['pose']
        if fmt =='xyzrpy':
            curr_pose = pose_xyzrpy
        elif fmt == 'matrix':
            curr_pose = np.eye(4)
            curr_pose[:3, :3] = Rot.from_euler('xyz', pose_xyzrpy[3:]).as_matrix()
            curr_pose[:3, 3] = pose_xyzrpy[0:3]
        else:
            print(f'Unsupported fmt: {fmt}, should be \'xyzrpy\' or \'matrix\'')
            return None, None
        return curr_q, curr_pose

    def moveJ(self, joint, v=20, r=0, connect=0, block=1, degrees=False) -> None:
        if not degrees:
            joint = [jq * 180 / np.pi for jq in joint]
        movej_result = self.robot.rm_movej(joint, v, r, connect, block)
        if movej_result == 0:
            print("\nmovej motion succeeded\n")
        else:
            print("\nmovej motion failed, Error code: ", movej_result, "\n")

    def movej_p(self, pose, v=20, r=0, connect=0, block=1):
        movej_p_result = self.robot.rm_movej_p(pose, v, r, connect, block)
        if movej_p_result == 0:
            print("\nmovej_p motion succeeded\n")
        else:
            print("\nmovej_p motion failed, Error code: ", movej_p_result, "\n")

    def moves(self, move_positions=None, speed=20, blending_radius=0, block=1):
        if move_positions is None:
            move_positions = [
                [-0.3, 0, 0.3, 3.14, 0, 0],
                [-0.27, -0.22, 0.3, 3.14, 0, 0],
                [-0.314, -0.25, 0.2, 3.14, 0, 0],
                [-0.239, 0.166, 0.276, 3.14, 0, 0],
                [-0.239, 0.264, 0.126, 3.14, 0, 0]
            ]

        for i, pos in enumerate(move_positions):
            current_connect = 1 if i < len(move_positions) - 1 else 0
            moves_result = self.robot.rm_moves(pos, speed, blending_radius, current_connect, block)
            if moves_result != 0:
                print(f"\nmoves operation failed, error code: {moves_result}, at position: {pos}\n")
                return

        print("\nmoves operation succeeded\n")

    def moveL(self, pose, v=20, r=0, connect=0, block=1):
        """
        Perform movel motion.

        Args:
            pose (list of float): End position [x, y, z, rx, ry, rz].
            v (float, optional): Speed of the motion. Defaults to 20.
            connect (int, optional): Trajectory connection flag. Defaults to 0.
            block (int, optional): Whether the function is blocking (1 for blocking, 0 for non-blocking). Defaults to 1.
            r (float, optional): Blending radius. Defaults to 0.

        Returns:
            None
        """
        movel_result = self.robot.rm_movel(pose, v, r, connect, block)
        if movel_result == 0:
            print("\nmovel motion succeeded\n")
        else:
            print("\nmovel motion failed, Error code: ", movel_result, "\n")
        return movel_result

    def moveL_relative(self, pose: list[float],
                       v: int=20, 
                       r: int=0, 
                       connect: int=0, 
                       frame_type: int=0, 
                       block: int=1):
        return self.robot.rm_movel_offset(pose, v, r, connect, frame_type, block)


    def moveL_relative_v2(self, relative_motion,
                            v: int=20, 
                            r: int=0, 
                            connect: int=0,
                            block: int=1):
        _, curr_pose = self.get_current_state(fmt='matrix')
        target_pose_mat = curr_pose @ relative_motion
        target_pose_cmd = self.pose_mat2vec(target_pose_mat)
        return self.moveL(target_pose_cmd, v, r, connect, block)

    def servoJ(self, joint: list[float]):
        return self.robot.rm_movej_canfd(joint, False)

    def move_to_pose(self, target_pose, v=20, r=0, connect=0, block=1):
        pose = self.pose_mat2vec(target_pose)
        return self.moveL(pose, v, r, connect, block)

    def moveJ_relative(self, target_q_inc, spd, block):
        curr_q = self.get_joint_state()
        target_q = [q1 + q2 for q1, q2 in zip(curr_q, target_q_inc)]
        return self.moveJ(target_q, v=spd, block=block)

def main():
    # Create a robot arm controller instance and connect to the robot arm
    robot_controller = RobotArmController("192.168.1.18", 8080, 3)

    # Get API version
    print("\nAPI Version: ", rm_api_version(), "\n")

    arm_model = robot_controller.get_arm_model()
    points = arm_models_to_points.get(arm_model, [])

    # Perform movej_p motion
    robot_controller.moveJ(points[0])

    # Perform movej_p motion
    robot_controller.movej_p(points[1])

    # Perform move operations
    robot_controller.moves(points[2])

    # Move to target position [xyz, rx ry rz]
    robot_controller.moveL([0.2, 0, 0.4, 3.141, 0, 0], v=50)

    # Disconnect the robot arm
    robot_controller.disconnect()


if __name__ == "__main__":
    main()
