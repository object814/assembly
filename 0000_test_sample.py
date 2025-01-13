import cv2
import time
import viser 
import numpy as np
import open3d as o3d
import hydra
import torch
import warnings
from utils import PointCloudUtils
from xarm6_interface import XARM6_IP, XARM6LEFT_IP
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR+"/third_party/segment-anything"))
print(sys.path)
from xarm6_interface.utils.realsense import MultiRealsense, get_masked_pointcloud, remove_outliers
from pathlib import Path
from scipy.spatial.transform import Rotation as R

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
from xarm6_interface.arm_pk import XArm6WOEE
from xarm6_interface.utils.viser_utils import update_viser_mp_result
from third_party.FoundationPose.estimater import *
import pickle
from segment_anything import sam_model_registry, SamPredictor
from graspnetAPI import GraspNet
from sklearn.decomposition import PCA
from pdb import set_trace as bp
from RPCA import RPCA
from modify_PCA import MPCA

'''https://drive.google.com/drive/folders/1zLZKragiKt6E9KdHWmKDGTo7EvDj1y6w?usp=sharing'''
'''https://drive.google.com/file/d/1PAJjSJDCQtjZaPzrfrZhIfiiSFRKOCH8/view?usp=sharing'''
# def estimate_pose_with_icp(source_mesh, target_point_cloud):
#     # 将 mesh 转换为点云
#     source_point_cloud = source_mesh.sample_points_uniformly(number_of_points=1000)
    
#     # 初始对齐（单位变换矩阵）
#     initial_transform = np.eye(4)

#     # 使用 ICP 进行配准
#     reg_p2p = o3d.pipelines.registration.registration_icp(
#         source_point_cloud, target_point_cloud, max_correspondence_distance=0.02,
#         estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         init=initial_transform
#     )
    
#     return reg_p2p.transformation  # 返回姿态矩阵

# # 示例
# source_mesh = o3d.io.read_triangle_mesh("path_to_source_mesh.obj")
# target_pcd = o3d.io.read_point_cloud("path_to_target_point_cloud.ply")
# pose = estimate_pose_with_icp(source_mesh, target_pcd)
# print("Estimated Pose:", pose)
def heuristic_principal_axes(point_cloud):
    # 计算点云的质心
    center = np.mean(point_cloud, axis=0)
    # 计算每个点到质心的距离
    distances = np.linalg.norm(point_cloud - center, axis=1)
    # 找到离质心最远的点
    max_distance_index = np.argmax(distances)
    x_axis = point_cloud[max_distance_index] - center
    x_axis /= np.linalg.norm(x_axis)  # 归一化 x 轴向量
    
    # 将所有点投影到 yz 平面
    projected_cloud = point_cloud - np.outer(np.dot(point_cloud, x_axis), x_axis)
    # 计算投影后点到原点的距离
    projected_distances = np.linalg.norm(projected_cloud[:, 1:], axis=1)
    # 找到投影后离原点最远的点
    max_projected_index = np.argmax(projected_distances)
    y_axis = projected_cloud[max_projected_index]
    y_axis /= np.linalg.norm(y_axis)  # 归一化 y 轴向量
    
    # 根据右手系规则计算 z 轴，即 z_axis = x_axis × y_axis
    z_axis = np.cross(x_axis, y_axis)
    
    # 构建旋转矩阵
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    return rotation_matrix

def average_direction_principal_axes(point_cloud, k=1):
    # 计算点云的质心
    center = np.mean(point_cloud, axis=0)
    # 计算每个点到质心的距离
    distances = np.linalg.norm(point_cloud - center, axis=1)
    # 找到离质心最远的 k 个点的索引
    farthest_k_indices = np.argsort(distances)[-k:]
    # 计算最远的 k 个点和质心连线的方向
    directions = point_cloud[farthest_k_indices] - center
    # 计算平均方向作为 x 轴
    x_axis = np.mean(directions, axis=0)
    x_axis /= np.linalg.norm(x_axis)  # 归一化 x 轴向量
    
    # 将所有点投影到与 x 轴正交的平面
    projected_cloud = point_cloud - np.outer(np.dot(point_cloud, x_axis), x_axis)
    # 找到投影后的点中离质心最远的 k 个点的索引
    projected_distances = np.linalg.norm(projected_cloud - center, axis=1)
    projected_farthest_k_indices = np.argsort(projected_distances)[-k:]
    # 计算投影后的 k 个点和质心连线的方向作为 y 轴
    projected_directions = projected_cloud[projected_farthest_k_indices] - center
    y_axis = np.mean(projected_directions, axis=0)
    y_axis /= np.linalg.norm(y_axis)  # 归一化 y 轴向量
    
    # 根据右手系规则计算 z 轴，即 z_axis = x_axis × y_axis
    z_axis = np.cross(x_axis, y_axis)
    
    # 构建旋转矩阵
    rotation_matrix = np.array([x_axis, y_axis, z_axis]).T
    return rotation_matrix


def center_point_cloud(point_cloud):
    center = np.mean(point_cloud, axis=0)
    centered_cloud = point_cloud - center
    return centered_cloud, center


def canonicalize_point_cloud_heu(point_cloud):
    centered_cloud, center = center_point_cloud(point_cloud)
    # rotation_matrix = heuristic_principal_axes(centered_cloud)
    rotation_matrix = average_direction_principal_axes(centered_cloud, k=3)
    canonical_cloud = centered_cloud @ rotation_matrix
    # 转换为 Open3D 点云对象
    canonicalized_pcd = o3d.geometry.PointCloud()
    canonicalized_pcd.points = o3d.utility.Vector3dVector(canonical_cloud)
    return canonicalized_pcd, rotation_matrix.T, center

def sample_points_from_mesh(mesh, num_points, seed=None):
    if seed is not None:
        np.random.seed(seed)  # 设置随机数种子
    points, _ = trimesh.sample.sample_surface_even(mesh, num_points)
    return points


def random_rotation_matrix():
    """
    Generate a random 3x3 rotation matrix (R ∈ SO(3)).
    """
    # Generate a random 3x3 matrix
    random_matrix = np.random.randn(3, 3)
    
    # Perform QR decomposition to get an orthogonal matrix
    Q, R = np.linalg.qr(random_matrix)
    
    # Ensure the determinant is 1 to make it a valid rotation matrix
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    
    return Q

def canonicalize_point_cloud_with_longest_axis(point_pcd):
    # 检查输入类型
    if isinstance(point_pcd, o3d.geometry.PointCloud):
        points = np.asarray(point_pcd.points)
    elif isinstance(point_pcd, np.ndarray):
        points = point_pcd
    else:
        raise TypeError("Input must be a numpy.ndarray or an open3d.geometry.PointCloud.")

    # 计算点云中心
    centroid = np.mean(points, axis=0)
    centered_pcd = points - centroid

    # 计算点云的协方差矩阵，找到主方向
    cov_matrix = np.cov(centered_pcd, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

    # 找到最大特征值对应的特征向量（延展最长的方向）
    longest_axis_index = np.argmax(eig_vals)
    x_axis = eig_vecs[:, longest_axis_index]

    # 确保 x 轴方向的一致性（可选）
    if x_axis[0] < 0:
        x_axis = -x_axis

    # 找到其他两个正交轴
    y_axis = eig_vecs[:, (longest_axis_index + 1) % 3]  # 第二个特征向量
    z_axis = np.cross(x_axis, y_axis)  # 计算正交的 z 轴
    z_axis /= np.linalg.norm(z_axis)  # 单位化
    y_axis = np.cross(z_axis, x_axis)  # 重新计算 y 轴，确保正交
    y_axis /= np.linalg.norm(y_axis)

    # 构造旋转矩阵
    rotation_matrix = np.stack([x_axis, y_axis, z_axis], axis=0)

    # 确保旋转矩阵的行列式为正（右手坐标系）
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix[2, :] = -rotation_matrix[2, :]

    # 将点云应用旋转矩阵
    transformed_pcd = np.dot(centered_pcd, rotation_matrix.T)

    # 转换为 Open3D 点云对象
    canonicalized_pcd = o3d.geometry.PointCloud()
    canonicalized_pcd.points = o3d.utility.Vector3dVector(transformed_pcd)

    return canonicalized_pcd, rotation_matrix, centroid

def robust_canonicalize_point_cloud(point_cloud):
    # 计算点云的质心
    center = np.mean(point_cloud, axis=0)
    centered_cloud = point_cloud - center
    # 计算协方差矩阵
    cov_matrix = np.cov(centered_cloud.T)
    # 计算协方差矩阵的特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    # 找到最大特征值对应的特征向量
    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvector = eigenvectors[:, max_eigenvalue_index]
    # 确保最大特征向量的第一个元素为正
    if max_eigenvector[0] < 0:
        max_eigenvector = -max_eigenvector
    # 构建旋转矩阵，将最大特征向量与 x 轴对齐
    x_axis = np.array([1, 0, 0])
    rotation_axis = np.cross(max_eigenvector, x_axis)
    cos_theta = np.dot(max_eigenvector, x_axis)
    sin_theta = np.linalg.norm(rotation_axis)
    if sin_theta == 0:
        rotation_matrix = np.eye(3)
    else:
        rotation_axis /= sin_theta
        skew_matrix = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                             [rotation_axis[2], 0, -rotation_axis[0]],
                             [-rotation_axis[1], rotation_axis[0], 0]])
        rotation_matrix = np.identity(3) + sin_theta * skew_matrix + (1 - cos_theta) * np.dot(skew_matrix, skew_matrix)
    # 旋转点云
    canonical_cloud = centered_cloud @ rotation_matrix.T
    return canonical_cloud, rotation_matrix, center

def canonicalize_point_cloud(point_pcd):
    if isinstance(point_pcd, o3d.geometry.PointCloud):
        points = np.asarray(point_pcd.points)
    elif isinstance(point_pcd, np.ndarray):
        points = point_pcd
    else:
        raise TypeError("Input must be a numpy.ndarray or an open3d.geometry.PointCloud.")

    '''get center'''
    # points = np.asarray(point_pcd.points)
    centroid = np.mean(points, axis=0)
    centered_pcd = points - centroid
    # 3. 使用 PCA 计算主轴方向

    
    pca = PCA(n_components=3)
    pca.fit(centered_pcd)
    # 4. 获取旋转矩阵（主轴方向）
    rotation_matrix = pca.components_
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix = -rotation_matrix
    # print(f"rotation_matrix: {rotatisample_surfaceon_matrix}")
    transformed_pcd = np.dot(centered_pcd, rotation_matrix.T)
    # restored_pcd = centered_pcd + centroid

    # 将还原的点云转换回 Open3D 点云对象
    canonicalized_pcd = o3d.geometry.PointCloud()
    canonicalized_pcd.points = o3d.utility.Vector3dVector(transformed_pcd)

    return canonicalized_pcd, rotation_matrix, centroid

def canonicalize_point_cloud_with_bounding_box(point_pcd):
    # 检查输入类型
    if isinstance(point_pcd, o3d.geometry.PointCloud):
        points = np.asarray(point_pcd.points)
    elif isinstance(point_pcd, np.ndarray):
        points = point_pcd
    else:
        raise TypeError("Input must be a numpy.ndarray or an open3d.geometry.PointCloud.")

    # 计算点云的包围盒
    oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))

    # 获取包围盒的中心和旋转矩阵
    centroid = np.array(oriented_bbox.center)
    rotation_matrix = np.array(oriented_bbox.R)
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix = -rotation_matrix

    # 将点云应用包围盒的旋转矩阵进行对齐
    centered_pcd = points - centroid  # 将点云平移到包围盒中心
    transformed_pcd = np.dot(centered_pcd, rotation_matrix)  # 应用旋转矩阵

    # 转换为 Open3D 点云对象
    canonicalized_pcd = o3d.geometry.PointCloud()
    canonicalized_pcd.points = o3d.utility.Vector3dVector(transformed_pcd)

    return canonicalized_pcd, rotation_matrix.T, centroid

def canonicalize_point_cloud_with_icp(point_pcd):
    """
    Canonicalize a point cloud by aligning it to its oriented bounding box using ICP.

    Parameters:
        point_pcd (o3d.geometry.PointCloud or np.ndarray): Input point cloud.

    Returns:
        canonicalized_pcd (o3d.geometry.PointCloud): Canonicalized point cloud.
        rotation_matrix (np.ndarray): Rotation matrix aligning the point cloud.
        centroid (np.ndarray): Centroid of the point cloud.
    """
    # 检查输入类型
    if isinstance(point_pcd, o3d.geometry.PointCloud):
        points = np.asarray(point_pcd.points)
    elif isinstance(point_pcd, np.ndarray):
        points = point_pcd
    else:
        raise TypeError("Input must be a numpy.ndarray or an open3d.geometry.PointCloud.")

    # Convert to Open3D PointCloud
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(points)

    # 计算点云的包围盒
    oriented_bbox = o3d.geometry.OrientedBoundingBox.create_from_points(o3d.utility.Vector3dVector(points))
    bbox_points = np.asarray(oriented_bbox.get_box_points())

    # Create target point cloud from bounding box corners
    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(bbox_points)

    # 初始变换矩阵
    initial_transform = np.eye(4)

    # 使用 ICP 进行配准
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pcd, target_pcd, max_correspondence_distance=0.02,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        init=initial_transform
    )

    # 提取变换矩阵
    transformation = reg_p2p.transformation
    rotation_matrix = transformation[:3, :3]
    centroid = transformation[:3, 3]

    # 将点云应用变换矩阵
    canonicalized_pcd = source_pcd.transform(transformation)

    return canonicalized_pcd, rotation_matrix.T, centroid
def test():
    pass
    
if __name__ == "__main__":
    sv = viser.ViserServer()
    Rpca = RPCA()
    Mpca = MPCA()
    random_R_list = []
    random_t_list = []
    for i in range(10):
        random_R = random_rotation_matrix()
        random_t = np.random.rand(3)
        random_R_list.append(random_R)
        random_t_list.append(random_t)

    # mesh_dir = "object_mesh_new/board00/board00.obj"
    # mesh_dir = "object_mesh_new/original_part_00/original_part_00.obj"
    # mesh = trimesh.load_mesh(mesh_dir)
    # pts_0= sample_points_from_mesh(mesh, 10000)
    for i in range(10):
        # pts = sample_points_from_mesh(mesh, 10000)
        pts = np.load("pts2.npy")
        random_R = random_R_list[i]
        random_t = random_t_list[i]
        new_point_cloud = o3d.geometry.PointCloud()
        rotated_pcd = pts @ random_R + random_t
        new_point_cloud.points = o3d.utility.Vector3dVector(rotated_pcd)
        # canonicalized_pcd, rotation_matrix0, centroid0 = canonicalize_point_cloud_heu(np.asarray(new_point_cloud.points))
        canonicalized_pcd, rotation_matrix0, centroid0 = PointCloudUtils.canonical_bbo(new_point_cloud, visualize = False)
        print(np.linalg.det((rotation_matrix0)))
        rotation_matrix_writable = np.array(rotation_matrix0, copy=True)
        sv.scene.add_point_cloud(f"ori{i}",points = np.asarray(new_point_cloud.points),colors=(255,0,0),point_size=0.002,point_shape="circle")
        sv.scene.add_point_cloud(f"pts{i}",points =canonicalized_pcd,colors=(0,255,0),point_size=0.002,point_shape="circle")
        sv.scene.add_frame(f"PCA{i}", wxyz=R.from_matrix(rotation_matrix_writable[:3, :3].T).as_quat()[[3, 0, 1, 2]], position=centroid0, axes_length=0.3, axes_radius=0.01)
    input("Press Enter to continue...")