import numpy as np
import open3d as o3d
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
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR+"/third_party/segment-anything"))
print(sys.path)
from third_party.FoundationPose.estimater import *
import pickle
from segment_anything import sam_model_registry, SamPredictor
from graspnetAPI import GraspNet
from sklearn.decomposition import PCA
from pdb import set_trace as bp
from order_planing import order_planing

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import trimesh
import time

class PointCloudUtils:
    """
    点云工具类，包含点云处理、坐标变换、螺旋运动等功能。
    """

    @staticmethod
    def read_matrices_from_npy(file_path):
        data = np.load(file_path)
        data = data[0]
        matrices = [np.array(matrix) for matrix in data]
        return matrices

    @staticmethod
    def se3_distance(pose1, pose2):
        trans_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) ** 2
        return trans_diff

    @staticmethod
    def sample_points_from_mesh(mesh, num_points, seed=None):
        if seed is not None:
            np.random.seed(seed)
        points, _ = trimesh.sample.sample_surface(mesh, num_points)
        return points

    @staticmethod
    def extract_rectangle_with_pose(point_cloud):
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            point_cloud = np.asarray(point_cloud.points)
        elif not isinstance(point_cloud, np.ndarray):
            raise ValueError("Invalid type of point_cloud")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        rotation_matrix = obb.R
        lengths = obb.extent
        pcd_center = np.mean(point_cloud)
        vector1 = center-pcd_center
        if np.dot(rotation_matrix[:3,2],vector1)<0:
            rotation_matrix = rotation_matrix @ R.from_euler('x', 180, degrees=True).as_matrix()

            

        return center, rotation_matrix, lengths
    
    @staticmethod
    def canonical_bbo(point_cloud, visualize = False, reverse_xy = False):
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            point_cloud = np.asarray(point_cloud.points)
        elif not isinstance(point_cloud, np.ndarray):
            raise ValueError("Invalid type of point_cloud")

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        obb = pcd.get_oriented_bounding_box()
        center = obb.center
        rotation_matrix = obb.R
        if np.linalg.det(rotation_matrix)<0:
            rotation_matrix = -rotation_matrix
        lengths = obb.extent
        pcd_center = np.mean(point_cloud, axis = 0)
        vector1 = center-pcd_center
        if np.dot(rotation_matrix[:3,2],vector1)<0:
            # bp()
            rotation_matrix = rotation_matrix @ R.from_euler('x', 180, degrees=True).as_matrix()

        canonical_pcd = (point_cloud-pcd_center) @ rotation_matrix

        if reverse_xy:
            rotation = R.from_euler('z', 180, degrees = True).as_matrix()
            rotation_matrix = rotation_matrix @ rotation

        if visualize:
            # 原始点云
            original_pcd = o3d.geometry.PointCloud()
            original_pcd.points = o3d.utility.Vector3dVector(point_cloud)
            original_pcd.paint_uniform_color([0, 1, 0])  # 绿色

            # 规范化后的点云
            canonical_pcd_o3d = o3d.geometry.PointCloud()
            canonical_pcd_o3d.points = o3d.utility.Vector3dVector(canonical_pcd)
            canonical_pcd_o3d.paint_uniform_color([0, 0, 1])  # 蓝色

            # 包围盒
            obb.color = (1, 0, 0)  # 红色

            # 坐标系
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

            # 可视化
            o3d.visualization.draw_geometries(
                [original_pcd, obb, canonical_pcd_o3d, axes],
                window_name="Point Cloud with OBB and Canonical View",
                width=800, height=600
            )

        return canonical_pcd, rotation_matrix.T, pcd_center

    @staticmethod
    def canonicalize_point_cloud(point_cloud):
        points = np.asarray(point_cloud.points)
        centroid = np.mean(points, axis=0)
        centered_pcd = points - centroid
        pca = PCA(n_components=3)
        pca.fit(centered_pcd)
        rotation_matrix = pca.components_
        transformed_pcd = np.dot(centered_pcd, rotation_matrix.T)
        canonicalized_pcd = o3d.geometry.PointCloud()
        canonicalized_pcd.points = o3d.utility.Vector3dVector(transformed_pcd)
        return canonicalized_pcd, rotation_matrix, centroid

    @staticmethod
    def center_point_cloud(point_cloud):
        center = np.mean(point_cloud, axis=0)
        centered_cloud = point_cloud - center
        return centered_cloud, center

    @staticmethod
    def average_direction_principal_axes(point_cloud, k=3):
        center = np.mean(point_cloud, axis=0)
        distances = np.linalg.norm(point_cloud - center, axis=1)
        farthest_k_indices = np.argsort(distances)[-k:]
        directions = point_cloud[farthest_k_indices] - center
        x_axis = np.mean(directions, axis=0)
        x_axis /= np.linalg.norm(x_axis)
        projected_cloud = point_cloud - np.outer(np.dot(point_cloud, x_axis), x_axis)
        projected_distances = np.linalg.norm(projected_cloud - center, axis=1)
        projected_farthest_k_indices = np.argsort(projected_distances)[-k:]
        projected_directions = projected_cloud[projected_farthest_k_indices] - center
        y_axis = np.mean(projected_directions, axis=0)
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
        return rotation_matrix

    @staticmethod
    def canonicalize_point_cloud_heu(point_cloud):
        if isinstance(point_cloud, o3d.geometry.PointCloud):
            point_cloud = np.asarray(point_cloud.points)
        elif not isinstance(point_cloud, np.ndarray):
            raise ValueError("Invalid type of point_cloud")
        
        centered_cloud, center = PointCloudUtils.center_point_cloud(point_cloud)
        rotation_matrix = PointCloudUtils.average_direction_principal_axes(centered_cloud, k=3)
        canonical_cloud = centered_cloud @ rotation_matrix
        canonicalized_pcd = o3d.geometry.PointCloud()
        canonicalized_pcd.points = o3d.utility.Vector3dVector(canonical_cloud)
        return canonicalized_pcd, rotation_matrix.T, center

    @staticmethod
    def calculate_transform_matrix(rotation_path, center_path, base_pcd):
        base_pcd = np.asarray(base_pcd.points)
        rotation_mats = PointCloudUtils.read_matrices_from_npy(rotation_path)
        centers = PointCloudUtils.read_matrices_from_npy(center_path)
        base_rotation_cam  = rotation_mats[0]
        base_center_cam = centers[0]

        base_cam = np.eye(4)
        base_cam[:3, :3] = base_rotation_cam.T
        base_cam[:3, 3] = base_center_cam
        
        canonical_base_pcd, base_mat, base_center = PointCloudUtils.canonicalize_point_cloud_heu(base_pcd)
        canonical_base_mat = np.eye(4)
        canonical_base_mat[:3, :3] = base_mat.T
        canonical_base_mat[:3, 3] = base_center
        
        canonical_transform_base_cam = canonical_base_mat @ np.linalg.inv(base_cam) 
        return canonical_transform_base_cam, rotation_mats, centers

    @staticmethod
    def spiral_motion_dynamic_x_axis(
        xarm, initial_radius=10, final_radius=2, pitch=2, loops=3, steps_per_loop=20,
        speed=50, mvacc=500, gripper_open=850, gripper_close=600, control_frequency=50
    ):
        total_steps = loops * steps_per_loop
        step_time = 1.0 / control_frequency
        start_time = time.time()

        for i in range(total_steps):
            theta = 2 * np.pi * (i / steps_per_loop)
            radius = initial_radius + (final_radius - initial_radius) * (i / total_steps)
            y_offset = radius * np.sin(theta)
            z_offset = radius * np.cos(theta) + pitch * (i / steps_per_loop)

            _, current_pose = xarm.get_position_se3(is_radian=False)
            current_position = current_pose[:3, 3]
            current_rotation = current_pose[:3, :3]
            rotation = R.from_matrix(current_rotation)
            r, p, y = rotation.as_euler("XYZ", degrees=False)
            tool_offset = np.array([0, y_offset, z_offset])
            world_target = current_position + current_rotation @ tool_offset

            xarm.set_tool_position(
                x=world_target[0], y=world_target[1], z=world_target[2],
                roll=r, pitch=p, yaw=y,
                speed=speed, mvacc=mvacc, is_radian=False, wait=False
            )

            gripper_position = int(gripper_close + (gripper_open - gripper_close) * (i / total_steps))
            xarm.arm.set_gripper_position(gripper_position, wait=False)

            elapsed_time = time.time() - start_time
            sleep_time = step_time - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            start_time = time.time()

        xarm.arm.set_gripper_position(gripper_open, wait=True)
        print("螺旋运动完成，夹爪已松开。")
