import numpy as np
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR+"3rd_party/segment-anything"))
sys.path.append(os.path.join(ROOT_DIR+"3rd_party/xarm6"))
import cv2
import time
import viser 
import numpy as np
import open3d as o3d
import hydra
import torch
import warnings
from xarm6_interface import XARM6_IP, XARM6LEFT_IP
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
print(sys.path)
import pyrealsense2 as rs
from xarm6_interface.utils.realsense import MultiRealsense, get_masked_pointcloud, remove_outliers
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from xarm6_interface import XARM6_WO_EE_URDF_PATH
from xarm6_interface import SAM_TYPE, SAM_PATH
from xarm6_interface.utils.sam_prompt_drawer import SAMPromptDrawer
from xarm6_interface.arm_rw import XArm6RealWorld
from scipy.spatial.transform import Slerp
from xarm6_interface.arm_mplib import XARM6PlannerCfg, XARM6Planner, min_jerk_interpolator_with_alpha
import pytorch3d
from matplotlib import pyplot as plt
from loguru import logger as lgr
# from leaphand_rw.leaphand_rw import LeapNode, leap_from_sim_to_rw
from xarm6_interface.envs.table_and_workspace_pc import WoodenTableMount, create_bounding_box_pc, create_plane_pc, env_pc_post_process
from xarm6_interface.arm_pk import XArm6WOEE, RobotArm
from xarm6_interface.utils.viser_utils import update_viser_mp_result
from third_party.FoundationPose.estimater import *
import pickle
from segment_anything import sam_model_registry, SamPredictor
from graspnetAPI import GraspNet
from sklearn.decomposition import PCA
from pdb import set_trace as bp
from order_planing import order_planing
from utils import PointCloudUtils, Grasp_Policy
import sapien



class XArmController:
    def __init__(self, ip, planner_cfg=None):

        self.ip = ip
        self.arm = XArm6RealWorld(ip=ip)
        self.planner_cfg = planner_cfg or XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=1.0 / 50.0)
        self.planner = XARM6Planner(self.planner_cfg)
        grippermount_data_dir = Path("data/data_urdf/robot/xarm_gripper/hand_open_cvx_hull.obj")
        gripper_trimesh = trimesh.load_mesh(grippermount_data_dir)
        gripper_transform = np.eye(4)
        self.planner.mplib_update_attached_object(
            gripper_trimesh,
            gripper_transform,
            name = "gripper",
        )

    def move_to_home(self):

        self.arm.set_joint_values(self.arm.default_joint_values, speed=0.3, wait=True)
    
    def move_down(self, distance = 0.1):
        _, current_position = self.arm.get_position_se3()
        current_position = np.array(current_position)
        current_position[:3, 3] -= np.array([0, 0, distance])
        target_position = current_position.copy()
        target_position[:3, 3] *= 1000
        self.arm.set_position_from_matrix(target_position)

    def move_y_axis(self, distance = 0.1):
        _, current_position = self.arm.get_position_se3()
        current_position = np.array(current_position)
        offset_y =  current_position[:3,:3] @ np.array([0, distance, 0])
        target_position = current_position.copy()
        target_position[:3, 3] += offset_y
        target_position[:3, 3] *= 1000
        self.arm.set_position_from_matrix(target_position)

    def move_z_axis(self, distance = 0.1):
        _, current_position = self.arm.get_position_se3()
        current_position = np.array(current_position)
        offset_z =  current_position[:3,:3] @ np.array([0, 0, distance])
        target_position = current_position.copy()
        target_position[:3, 3] += offset_z
        target_position[:3, 3] *= 1000
        self.arm.set_position_from_matrix(target_position)

    def move_left(self, distance = 0.1):
        _, current_position = self.arm.get_position_se3()
        current_position = np.array(current_position)
        current_position[:3, 3] -= np.array([0, distance, 0])
        target_position = current_position.copy()
        target_position[:3, 3] *= 1000


        self.arm.set_position_from_matrix(target_position)

    def move_back(self, distance = 0.1):
        _, current_position = self.arm.get_position_se3()
        current_position = np.array(current_position)
        current_position[:3, 3] -= np.array([distance, 0, 0])
        target_position = current_position.copy()
        target_position[:3, 3] *= 1000


        self.arm.set_position_from_matrix(target_position)

    def move_foward(self, distance = 0.1):
        _, current_position = self.arm.get_position_se3()
        current_position = np.array(current_position)
        current_position[:3, 3] -= np.array([distance, 0, 0])
        target_position = current_position.copy()
        target_position[:3, 3] *= 1000

        self.arm.set_position_from_matrix(target_position)


    def plan_and_execute(self, target_pose,z_axis_offset = None, y_axis_offset = None, x_axis_offset = None):
        if self.ip == XARM6LEFT_IP:
            sv = sv_left
        else:
            sv = sv_right
        if z_axis_offset is not None:
            z_axis_offset = np.array([0, 0, z_axis_offset])
            target_pose[:3, 3] -= target_pose[:3, :3] @ z_axis_offset
        if y_axis_offset is not None:
            y_axis_offset = np.array([0, y_axis_offset, 0])
            target_pose[:3, 3] -= target_pose[:3, :3] @ y_axis_offset
        if x_axis_offset is not None:
            x_axis_offset = np.array([x_axis_offset, 0, 0])
            target_pose[:3, 3] -= target_pose[:3, :3] @ x_axis_offset

        current_joint_values = self.arm.get_joint_values()
        planning_result = self.planner.mplib_plan_pose(current_joint_values, target_pose)
        if planning_result['status'] != 'Success':
            lgr.info(f"Collision-free planning: Fail")
            time.sleep(5)
            return False
        lgr.info(f"Collision-free planning: Success")
        waypt_joint_values_np = planning_result['position']
        end_joint_values = waypt_joint_values_np[-1]
        update_viser_mp_result(sv, xarm6_pk, current_joint_values, end_joint_values, waypt_joint_values_np)
        sv.add_mesh_simple("hand_open", vertices=gripper_trimesh.vertices, faces=gripper_trimesh.faces, wxyz=R.from_matrix(target_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=target_pose[:3, 3], opacity=0.5)
    
        lgr.info("Please validate the plan in the Viser GUI.")
        validated = False
        validate_button = sv.gui.add_button("Execute",)
        # turn validated to True]
        def validate_true():
            nonlocal validated
            validated = True
            lgr.info("validated")
        validate_button.on_click(lambda _: validate_true())
        while True:
            time.sleep(0.2)
            if validated:
                break
        validated = False
        waypoints = planning_result['position']
        self.arm.set_joint_values_sequence(waypoints, planning_timestep=self.planner_cfg.timestep)
        self.arm.set_joint_values(waypoints[-1], speed=0.35, wait=True)

        return True

    def move_gripper(self, open = None, close = None, position = None, wait = True):
        if  position is not None:
            self.arm.arm.set_gripper_position(position, wait=wait)
        elif open is not None:
            self.arm.arm.set_gripper_position(850, wait=wait)
        elif close is not None:
            self.arm.arm.set_gripper_position(-10, wait=wait)
        else:
            raise ValueError("Please specify open, close or position")
        
    def move_by_joints_value(self, joints_value, speed = 0.35, wait = True):
        self.arm.set_joint_values(joints_value, speed=speed, wait=wait)
    
    def rotation(self, angle, wait = True):
        current_joint_values = self.arm.get_joint_values()
        current_joint_values = np.array(current_joint_values)*180/np.pi
        current_joint_values[5] += angle
        taregt_joint_values = current_joint_values*np.pi/180
        self.arm.set_joint_values(taregt_joint_values, speed=0.35, wait=True)
    
    def update_collision_pcd(self, pcd, name = None):
        if self.ip == XARM6LEFT_IP:
            sv = sv_left
        else:
            sv = sv_right

        if isinstance(pcd, o3d.geometry.PointCloud):
            pcd = np.asarray(pcd.points)
        elif not isinstance(pcd, np.ndarray):
            raise ValueError("Invalid pcd type")
        
        self.planner.mplib_add_point_cloud(pcd, name)
        print(f'center of the pcd is {np.mean(pcd, axis=0)}')
        sv.scene.add_point_cloud("collison_pcd", points=pcd, colors=(255, 0, 0), point_size=0.002, point_shape="circle")
    
    def remove_collision_pcd(self, name = None):

        self.planner.mplib_remove_point_cloud(name)
    
    def remove_attachment(self, name = None):
        self.planner.mplib_detach_object(name)


    def update_attach(self, object_name, pose):
        if self.ip == XARM6LEFT_IP:
            sv = sv_left
        else:
            sv = sv_right
        _, pose_frank = self.arm.get_position()
        mesh_path = f"object_mesh_new/{object_name}/{object_name}.obj"
        mesh = trimesh.load_mesh(mesh_path).apply_scale(0.9)
        pose_mesh = np.linalg.inv(pose_frank) @ pose
        self.planner.mplib_update_attached_object(
            mesh,
            pose_mesh,
            name = object_name
        )

def screw_search(search_depth = 0.0001, search_step = 0.0001):
    pass

if __name__ == '__main__':
    grippermount_data_dir = Path("data/data_urdf/robot/xarm_gripper/hand_open.obj")
    gripper_trimesh = trimesh.load_mesh(grippermount_data_dir).apply_scale(1.1)
    xarm6_pk = XArm6WOEE()
    xarm_left = XArmController(ip=XARM6LEFT_IP)
    xarm_right = XArmController(ip=XARM6_IP)
    sv_left = viser.ViserServer()
    sv_right = viser.ViserServer()
    x = np.linspace(-0., 0.05, 10)
    y = np.linspace(-0., 0.05, 10)
    # xarm_left.move_to_home()
    tolerance = 5
    move_down = 0.002
    while True:
        torques_og = xarm_left.arm.arm.get_joints_torque()
        print(torques_og)
        for i in range(10):
            for j in range(10):
                x_offset = 0.001 * (-1) ** i
                xarm_left.move_foward(x_offset)
                time.sleep(0.5)
                position_og = xarm_left.arm.get_position_se3()
                #print(position_og)
                xarm_left.move_down(move_down)
                time.sleep(0.5)
                position_after = xarm_left.arm.get_position_se3()
                #print(position_after)
                toruqes = xarm_left.arm.arm.get_joints_torque()
                print(toruqes)
                time.sleep(0.5)
                
                
                #diff = position_og[1][2,3]- position_after[1][2,3]
                #print(diff)

                #if abs(diff)<0.000712448:
                #    xarm_left.move_down(-move_down)
                #    time.sleep(0.1)
                #    print("move up")
                #else:
                #    print("move down")


                if np.max(np.abs(np.array(torques_og[1]) - np.array(toruqes[1]))) > tolerance:
                    xarm_left.move_down(-0.002)
                    time.sleep(0.5)
            y_offset = 0.001
            xarm_left.move_left(y_offset)
            time.sleep(0.5)

            

