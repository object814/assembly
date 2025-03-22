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


def read_matrices_from_npy(file_path):

    # Load data from .npy file
    data = np.load(file_path)
    # Check the structure of the loaded data
    print(f"Data shape: {data.shape}")
    data = data[0]
    # Convert to a list of matrices
    matrices = [np.array(matrix) for matrix in data]
    print(f"Number of matrices: {len(matrices)}")
    print(f" matrix shape: {matrices[0].shape}")

    return matrices

def se3_distance(pose1, pose2):
    """
    计算两个 SE(3) 矩阵之间的平移和旋转距离。
    
    Parameters:
    pose1, pose2: numpy.ndarray
        4x4 的变换矩阵,分别表示两个 SE(3) 位姿。
    
    Returns:
    float
        平移和旋转的综合距离。
    """
    # 平移分量的平方距离
    trans_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) ** 2

    # # 计算旋转分量的测地线距离
    # R1 = pose1[:3, :3]  # 提取 pose1 的旋转矩阵
    # R2 = pose2[:3, :3]  # 提取 pose2 的旋转矩阵
    # relative_rotation = np.dot(R1.T, R2)  # 计算相对旋转矩阵
    # angle = np.arccos(
    #     (np.trace(relative_rotation) - 1) / 2
    # )  # 根据旋转矩阵的迹计算测地线距离（旋转角度)

    # # 防止浮点误差导致 acos 输入超出 [-1, 1]
    # angle = np.clip(angle, 0, np.pi)

    # 平移和旋转平方和的平方根
    return trans_diff

def enviroment_constraint(xarm6_planner_cfg):
    env_params = WoodenTableMount()
    workspace_pc = create_bounding_box_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    table_plane_pc = create_plane_pc(env_params.table_plane_xmin, env_params.table_plane_ymin, env_params.table_plane_zmin, env_params.table_plane_xmax, env_params.table_plane_ymax, env_params.table_plane_zmax, xarm6_planner_cfg.n_env_pc)
    workspace_xmin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmin, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax, env_params.ymin, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymax_pc = create_plane_pc(env_params.xmin, env_params.ymax, env_params.zmin, env_params.xmax, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    env_pc = np.concatenate([workspace_pc, table_plane_pc, workspace_xmin_pc, workspace_ymin_pc, workspace_ymax_pc], axis=0)
    env_pc = env_pc_post_process(env_pc, filter_norm_thresh=0.1, n_save_pc=None)
    return env_pc

def sample_points_from_mesh(mesh, num_points, seed=None):
    if seed is not None:
        np.random.seed(seed)  # 设置随机数种子
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points

def get_pcd(object_name):
    mesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj")

# 使用PCA对批量点云进行规范化
def canonicalize_point_cloud(point_cloud):

    '''get center'''
    points = np.asarray(point_cloud.points)
    centroid = np.mean(points, axis=0)
    centered_pcd = points - centroid
    # 3. 使用 PCA 计算主轴方向
    pca = PCA(n_components=3)
    pca.fit(centered_pcd)
    # 4. 获取旋转矩阵（主轴方向)
    rotation_matrix = pca.components_
    print(f"rotation_matrix: {rotation_matrix}")
    transformed_pcd = np.dot(centered_pcd, rotation_matrix.T)
    # restored_pcd = centered_pcd + centroid

    # 将还原的点云转换回 Open3D 点云对象
    canonicalized_pcd = o3d.geometry.PointCloud()
    canonicalized_pcd.points = o3d.utility.Vector3dVector(transformed_pcd)
    return canonicalized_pcd, rotation_matrix, centroid

def get_object_pc_fp(object_name, arm_ip=XARM6_IP):
    global cam2leftbase, cam2rightbase, cam_serial, cam_K
    global initial, mask_list, bbox_list  # for order_planing
    object_name_dino = object_name.replace("_", " ") + "."
    mesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj")
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    arm_cam_K = cam_K
    if arm_ip == XARM6LEFT_IP:
        arm_cam_X_BaseCamera = cam2leftbase
    elif arm_ip == XARM6_IP:
        arm_cam_X_BaseCamera = cam2rightbase 
    else:
        raise ValueError("Invalid arm_ip")    
    multi_rs = MultiRealsense([cam_serial])
    multi_rs.set_intrinsics(0, arm_cam_K[0, 0], arm_cam_K[1, 1], arm_cam_K[0, 2], arm_cam_K[1, 2])
    for _ in range(50):
        multi_rs.getCurrentData()
        rtr_dict_list = multi_rs.getCurrentData()
    for rtr_dict in rtr_dict_list:
        rgb, depth = rtr_dict["rgb"], (rtr_dict["depth"].astype(np.float32) / 1000).astype(np.float32)  # rgb: np.array
        
        # # Method 1: manually select the object
        prompt_drawer.reset()
        manual_mask_np = prompt_drawer.run(rgb)  # mask_np: binary mask, True for object, False for background, (720, 1280)

        # Method 2: upstream provide the mask
        # Note that the manual image should be provided in ./order_planing/img/manual.png
        # if initial:
        #     plt.imsave(os.path.join(os.path.dirname(__file__), "order_planing", "img", "input.png"), rgb)
        #     mask_list, bbox_list = order_planing()
        #     initial = False
        #     # 针对凳子的特判
        #     mask_list = [mask_list[i] for i in [0, 2, 3, 1, 4]]
        # mask_np = mask_list.pop(0)  # (720, 1280)
        # # input(f"xor mask: {np.sum(manual_mask_np ^ mask_np)}, and mask: {np.sum(manual_mask_np & mask_np)}")
        # bbox = bbox_list.pop(0)
        # print(f"bbox: {bbox}\nlen(mask_list) remained: {len(mask_list)}")
        pose = est.register(K=arm_cam_K, rgb=rgb, depth=depth, ob_mask=manual_mask_np, iteration=10 if arm_ip == XARM6LEFT_IP else 20)
        pose = arm_cam_X_BaseCamera @ pose

        points = sample_points_from_mesh(mesh, 50000, seed=0)

        object_pc_o3d = o3d.geometry.PointCloud()
        object_pc_o3d.points = o3d.utility.Vector3dVector(points)
        object_pc_o3d.transform(pose)

    return object_pc_o3d, pose


def get_inter_rot(batch_size):
    top_down_rot = R.from_euler('Y', 90, degrees=True)
    side_rot = R.from_euler('X', 90, degrees=True)
    slerp = Slerp([0, 1], R.concatenate([top_down_rot, side_rot]))
    inter_times = np.linspace(0, 1, batch_size)
    inter_rots = slerp(inter_times)
    inter_rots = inter_rots.as_euler("XYZ", degrees=False)
    return inter_rots
    
L_fc_weight: float = 1.0
L_distance_weight: float = 600.0
L_self_pen_weight: float = 5000.0
L_object_pen_weight: float = 20000.0
L_joint_limit_weight: float = 10000.0
L_joint_ref_weight: float = 100.0

def get_closest_joint_value(current_joint_value, target_joint_values):
    min_diff = 1000000
    closest_target_joint_value = None
    for target_joint_value in target_joint_values:
        diff_list = np.abs(current_joint_value - target_joint_value)
        diff = diff_list.sum()
        if diff < min_diff:
            min_diff = diff
            closest_target_joint_value = target_joint_value
    
    return closest_target_joint_value

def get_pcd_center(pcd):
    return np.mean(np.asarray(pcd.points), axis=0)

import time
def spiral_motion_dynamic_x_axis(
    xarm, initial_radius=10, final_radius=2, pitch=2, loops=3, steps_per_loop=20,
    speed=50, mvacc=500, gripper_open=850, gripper_close=600, control_frequency=50
):
    """
    实现螺旋线绕机械臂动态工具点的 X 轴运动，工具点随着运动更新。
    """
    total_steps = loops * steps_per_loop  # 总步数
    step_time = 1.0 / control_frequency  # 控制周期

    # 初始化时间控制
    start_time = time.time()

    for i in range(total_steps):
        # 计算当前角度和半径
        theta = 2 * np.pi * (i / steps_per_loop)
        radius = initial_radius + (final_radius - initial_radius) * (i / total_steps)

        # 工具坐标系中的偏移量
        y_offset = radius * np.sin(theta)
        z_offset = radius * np.cos(theta) + pitch * (i / steps_per_loop)

        # 获取当前末端工具的位姿
        _, current_pose = xarm.get_position_se3(is_radian=False)

        # 提取位置部分
        current_position = current_pose[:3, 3] # 4x4 矩阵中的位移

        # 提取旋转矩阵部分
        current_rotation = current_pose[:3, :3]  # 4x4 矩阵中的旋转矩阵
        rotation = R.from_matrix(current_rotation)
        r, p, y = rotation.as_euler("XYZ", degrees=False)  # 提取当前姿态的欧拉角
        # 在工具坐标系下计算目标点
        tool_offset = np.array([0, y_offset, z_offset])  # 偏移量 (x 固定绕 y-z 平面)
        world_target = current_position + current_rotation @ tool_offset  # 转换到世界坐标系

        # 更新目标位置
        xarm.set_tool_position(
            x=world_target[0], y=world_target[1], z=world_target[2],
            roll=r, pitch=p, yaw=y,
            speed=speed, mvacc=mvacc, is_radian=False, wait=False
        )

        # 动态调整夹爪位置
        gripper_position = int(gripper_close + (gripper_open - gripper_close) * (i / total_steps))
        xarm.arm.set_gripper_position(gripper_position, wait=False)

        # 控制频率
        elapsed_time = time.time() - start_time
        sleep_time = step_time - elapsed_time
        if sleep_time > 0:
            time.sleep(sleep_time)
        start_time = time.time()  # 更新下一步的开始时间

    # 确保夹爪完全松开
    xarm.arm.set_gripper_position(gripper_open, wait=True)

    lgr.info("螺旋运动完成，夹爪已松开。")
    

def average_direction_principal_axes(point_cloud, k=3):
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
    
    # 根据右手系规则计算 z 轴,即 z_axis = x_axis × y_axis
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


planner_timestep = 1.0 / 50.0
cmd_timestep = 1.0 / 100.0 
pregrasp_retreat_distance = 0.08
# @hydra.main(version_base="1.2", config_path="", config_name="validate")

def calculate_transform_matrix(rotation_path, center_path, base_pcd):
        base_pcd = np.asarray(base_pcd.points)
        rotation_mats = read_matrices_from_npy(rotation_path)
        centers = read_matrices_from_npy(center_path)
        base_rotation_cam  = rotation_mats[0]
        base_center_cam = centers[0]

        base_cam = np.eye(4)
        base_cam[:3, :3] = base_rotation_cam.T
        base_cam[:3, 3] = base_center_cam
        canonical_base_pcd, base_mat, base_center = canonicalize_point_cloud_heu(base_pcd)
        canonical_base_mat = np.eye(4)
        canonical_base_mat[:3, :3] = base_mat.T
        canonical_base_mat[:3, 3] = base_center
        canonical_transform_base_cam = canonical_base_mat @ np.linalg.inv(base_cam) 

        sv.scene.add_frame("canonical_base_pose", wxyz=R.from_matrix(canonical_base_mat[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical_base_mat[:3, 3], axes_length=0.3, axes_radius=0.01)

        return canonical_transform_base_cam, rotation_mats, centers

def pick(object_name='sticker', arm_ip=XARM6_IP):  
    # object_pc_o3d, masked_pc_o3d_fusion = get_object_pc()
    t1 = time.time()

    def pose_reevaluate_and_attach():
        pass

    def pick_policy_z_axis():
        '''policy1 is to pick along the z axis of the object pose get from foundation psoe with 30cm pregrasp distance'''
        adjustment_rotation = R.from_euler('x', 90, degrees=True).as_matrix()
        # Embed the adjustment rotation into a 4x4 transformation matrix
        adjustment_rotation_4x4 = np.eye(4)
        adjustment_rotation_4x4[:3, :3] = adjustment_rotation
        adjusted_X_WorldObject = X_WorldObject @ adjustment_rotation_4x4
        adjusted_rotation_matrix = adjusted_X_WorldObject[:3, :3]

        # to see the origin pose
        wxyz = R.from_matrix(adjusted_rotation_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]  # 转换为 wxyz 格式
        position = X_WorldObject[:3, 3]

        # 偏移量在物体坐标系下
        offset_in_object_frame = np.array([0, 0, -0.3])  # 物体局部坐标系下的Z轴上方30cm

        # 将偏移量从物体坐标系转换到世界坐标系
        offset_in_world_frame = adjusted_rotation_matrix @ offset_in_object_frame

        center_grasp = np.eye(4,4)
        center_grasp[:3, 3] = pcd_center+offset_in_world_frame
        center_grasp[:3, :3] = adjusted_rotation_matrix
        # center_grasp[2,3] += 0.3
        policy = "z_axis"

        print(f"center_grasp: {center_grasp}")
        return center_grasp, adjusted_rotation_matrix, policy
    
    def pick_policy_top_down(pose, pcd_center):
        # Ensure z points down
        pose[:3, 2] = np.array([0, 0, -1])
        x_axis = pose[:3, 0]
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross([0,0,-1], x_axis)
        y_axis /= np.linalg.norm(y_axis)  # Normalize y

        pose[:3, 1] = y_axis

        offset_in_world_frame = np.array([0, 0, 0.3])
        pre_grasp = pose.copy()
        pre_grasp[:3, 3] = pcd_center + offset_in_world_frame  
        policy = "top_down"
        
        return pre_grasp, pose, policy
    
    def update_attach(object_name='sticker', pose_tool=None, pose_frank = None):
            grippermount_data_dir = Path("data/data_urdf/robot/xarm_gripper/hand_open_cvx_hull.obj")
            gripper_trimesh = trimesh.load_mesh(grippermount_data_dir).apply_scale(1.1)
            position = pose_tool[:3, 3]
            rotation_matrix = pose_tool[:3, :3]
            wxyz = R.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]  # Convert to wxyz format
            xarm6_planner.mplib_update_attached_object(
                gripper_trimesh,
                pose_frank,
            )
            sv.scene.add_mesh_simple(
                "attached_gripper",
                vertices=gripper_trimesh.vertices,
                faces=gripper_trimesh.faces,
                wxyz=wxyz,
                position=position,
                opacity=0.5,
            )
            #####################################
            
            ####zjx, attach trimesh
            attached_trimesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj").apply_scale(1.2)
            # attached_trimesh_cvx_hull = attached_trimesh.convex_hull
            # Create a transformation matrix for attaching (identity for testing)
            
            attach_transform = pose_tool  # 4x4 transformation matrix, identity for testing

            # Get position and rotation from the transformation matrix
            position = attach_transform[:3, 3]
            rotation_matrix = attach_transform[:3, :3]
            wxyz = R.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]  # Convert to wxyz format

            # Update the planner with the attached object
            xarm6_planner.mplib_update_attached_object(attached_trimesh, attach_transform)
    
    
    
 
    ''' setup the planner and vis '''
    # sv = viser.ViserServer()
    xarm6_pk = XArm6WOEE()
    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    env_pc = enviroment_constraint(xarm6_planner_cfg)
    xarm6_planner.mplib_add_point_cloud(env_pc, name="env_pc")
    ''' setup the planner and vis '''

    xarm = XArm6RealWorld(ip = arm_ip)
    xarm.arm.set_gripper_position(850, wait=True)
    home_joint_values = xarm.default_joint_values  # 默认的回到初始位置的关节角
    xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)

    object_pc_o3d, X_WorldObject = get_object_pc_fp(object_name, arm_ip = arm_ip)
    canonicalized_pcd, rotation_matrice, centroid = canonicalize_point_cloud(object_pc_o3d)


    sv.scene.add_point_cloud("canonical_pc", points=np.asarray(canonicalized_pcd.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    sv.scene.add_frame("canonical_pose", wxyz=R.from_matrix(rotation_matrice[:3, :3]).as_quat()[[3, 0, 1, 2]], position=centroid, axes_length=0.03, axes_radius=0.001)
    pcd_center  = get_pcd_center(object_pc_o3d)
    pre_grasp, adjusted_rotation_matrix, policy = pick_policy_top_down(X_WorldObject, pcd_center)
    wxyz = R.from_matrix(adjusted_rotation_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]  # 转换为 wxyz 格式
    sv.scene.add_frame("obejct_pose", wxyz=wxyz, position=pcd_center, axes_length=0.03, axes_radius=0.001)
    sv.scene.add_point_cloud("object_pc", points=np.asarray(object_pc_o3d.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
    current_joint_values = np.array(xarm.get_joint_values())

    # REAL PALNNING IS HERE! 
    planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, pre_grasp)
    if planning_result['status'] != 'Success':
        lgr.info(f"Collision-free planning: Fail")
        return

    # wxyz = trimesh.transformations.quaternion_from_matrix(selected_X_WorldEE)
    # pos = selected_X_WorldEE[:3, 3]
    # sv.scene.add_frame(f"frame_pose_{i}", wxyz=wxyz, position=pos, axes_length=0.03, axes_radius=0.001)
    lgr.info(f"Collision-free planning: Success")

    waypt_joint_values_np = planning_result['position']
    end_joint_values = waypt_joint_values_np[-1]
    update_viser_mp_result(sv, xarm6_pk, current_joint_values, end_joint_values, waypt_joint_values_np)
    # sv.add_mesh_simple("hand_open", vertices=gripper_trimesh.vertices, faces=gripper_trimesh.faces, wxyz=R.from_matrix(pre_grasp[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pre_grasp[:3, 3], opacity=0.5)
    
    # validated = False
    # validate_button = sv.gui.add_button("Execute",)
    # # turn validated to True]
    # def validate_true():
    #     nonlocal validated
    #     validated = True
    #     lgr.info("validated")
    # validate_button.on_click(lambda _: validate_true())
    # while True:
    #     time.sleep(0.2)
    #     if validated:
    #         break
    # validated = False
    xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.35, wait=True)
    
    # input("Press Enter to continue...")
    
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    current_joint_values = np.array(xarm.get_joint_values()) # yiwen

    ''' grasping motion is here'''
    if arm_ip==XARM6_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.13]) 
    elif arm_ip==XARM6LEFT_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.1215]) # 物体局部坐标系下向下偏移13cm  
    if policy == "top_down":
        center_grasp = pre_grasp.copy()
        center_grasp[:3, 3] = pre_grasp[:3,3]- offset_grasp_in_object_frame
        print(f"center_grasp: {center_grasp}")
    
    elif policy=="z_axis":
        center_grasp = pre_grasp.copy()
        offset_grasp_in_world_frame = adjusted_rotation_matrix @ offset_grasp_in_object_frame
        center_grasp[:3, 3] = pre_grasp[:3,3] + offset_grasp_in_world_frame

        print(f"center_grasp: {center_grasp}")


    
    
    status, grasp_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, center_grasp)
    closest_grasp_arm_joint_values = get_closest_joint_value(current_joint_values, grasp_arm_joint_values)
    mp_is_success = status == 'Success'
    if not mp_is_success:
        lgr.info(f"Grasp planning: Fail")
    else:
        grasp_joint_values = closest_grasp_arm_joint_values
        xarm.arm.set_servo_angle(angle=xarm.to_list(grasp_joint_values), speed=0.2, wait=True, is_radian=True)
        xarm.arm.set_gripper_position(-10, wait=True)

        '''grasping success, lift it up'''
        current_joint_values = np.array(xarm.get_joint_values())
        X_WorldEeflift =  center_grasp.copy()
        X_WorldEeflift[:3, 3] += np.array([0, 0, 0.2])
        status, lifted_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, X_WorldEeflift)
        mp_is_success = status == 'Success'
        closest_lifted_arm_joint_value = get_closest_joint_value(current_joint_values, lifted_arm_joint_values)
        if not mp_is_success:
            lgr.info(f"Lift planning: Fail")
        else:
            lgr.info(f"Lift planning: Success")
            xarm.arm.set_servo_angle(angle=xarm.to_list(closest_lifted_arm_joint_value), speed=0.2, wait=True, is_radian=True)
            # time.sleep(2)
            # xarm.arm.set_gripper_position(850, wait=True)

            # input("Press Enter to continue...")
            
            # obstacle_pcd, pose = get_object_pc_fp(object_name='chair2')
            # print(f"current pose: {pose}")
            # sv.scene.add_frame("obstacle_pose", wxyz=R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pose[:3, 3], axes_length=0.03, axes_radius=0.001)
            # sv.scene.add_point_cloud("object_pc", points=np.asarray(obstacle_pcd.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
            # xarm6_planner.mplib_add_point_cloud(np.asarray(obstacle_pcd.points), name="obstacle_pc")

            go_home_duration = 2
            waypt_joint_values_np = []
            current_joint_values = np.array(xarm.get_joint_values())
            waypt_joint_values_np.append(current_joint_values)
            waypt_joint_values_np.append(xarm.default_joint_values)
            waypt_joint_values_np = np.array(waypt_joint_values_np)
            xarm.set_joint_values_sequence(waypt_joint_values_np, go_home_duration)
            xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.2, wait=True)
            # get_current_attach_pose_offset(object_name='sticker')

            # go_to_test(pose)
        status, ee_pose = xarm.get_position_se3()
        status2, frank_pose = xarm.get_position()
        # update_attach(object_name='sticker', pose_tool=ee_pose,pose_frank = frank_pose)
        # bp()


            

def pose_initialization(arm_ip = XARM6LEFT_IP, rotation_path = None , center_path = None):

    global collision_pcd
    '''here we get all the target pose based on the left arm base'''
    # sv = viser.ViserServer()
    base_pcd, pose = get_object_pc_fp(object_name='original_part_00',arm_ip=arm_ip)
    collision_pcd = base_pcd
    sv.scene.add_point_cloud("base_pcd", points=np.asarray(base_pcd.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
    '''' get taget pose based on the canonical pose and assume the base in the real-world is static'''
    trans_mat, rotation_mats, centers= calculate_transform_matrix(rotation_path, center_path, base_pcd)
    
    for i in range(1,len(rotation_mats)):
        target_pose = np.eye(4)
        target_pose[:3, :3] = rotation_mats[i].T
        target_pose[:3, 3] = centers[i]
        target_pose = trans_mat @ target_pose
        target_list.append(target_pose)
        sv.scene.add_frame(f"target_pose{i}", wxyz=R.from_matrix(target_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=target_pose[:3, 3], axes_length=0.3, axes_radius=0.01)
    sorted_list = sorted(target_list, key = lambda x: x[0,3])
    global target_left, target_right
    target_left = sorted_list[:len(sorted_list)//2]
    target_right = sorted_list[len(sorted_list)//2:]
    # print(f"cam2leftbase: {cam2leftbase}")
    # print(f"cam2rightbase: {cam2rightbase}")
    mat_left_to_right = cam2rightbase@ np.linalg.inv(cam2leftbase)
    # print(f"mat_left_to_right: {mat_left_to_right}")
    target_right = [mat_left_to_right @ mat for mat in target_right]
    # print(f"target_right: {target_right}")


def insert(arm_ip=XARM6_IP, target_pose = None, mesh = None):
    global collision_pcd  # 声明使用全局变量
    maximun_planning_time = 10
 
    


    def get_pre_pose_from_target_pose(arm_ip, target_pose, ee_offset=0.166):
        ee_offset = ee_offset  # 末端执行器距离法兰的距离
        ee_offset_vector = np.array([0, 0, -ee_offset])  # 在目标坐标系下的偏移向量

        ee_target_pose = target_pose.copy()
        # 应用偏移
        ee_target_pose[:3, 3] += target_pose[:3, :3] @ ee_offset_vector.reshape(3)
        # ee_target_pose[:3, 3] -= ee_offset_grasp
        # 沿目标姿态的 x 轴偏移 
        if arm_ip ==  XARM6LEFT_IP:
            x_axis_offset = -0.011
        elif arm_ip == XARM6_IP:
            x_axis_offset = -0.02
        pre_align_pose = ee_target_pose.copy()
        x_offset_vector = np.array([x_axis_offset, 0, 0])  # 在目标坐标系下的 x 方向偏移向量
        pre_align_pose[:3, 3] += target_pose[:3, :3] @ x_offset_vector  # 应用偏移

        return pre_align_pose, ee_target_pose
    
    def update_collision_pcd(current_target_pcd):
        """
        Updates the collision point cloud by merging the current target point cloud
        into the existing collision point cloud in the scene.
        
        Args:
            current_target_pcd: The point cloud of the current target object.
        """
        global collision_pcd  # Ensure we update the global collision_pcd

        # Convert the existing collision_pcd to a numpy array
        existing_points = np.asarray(collision_pcd.points)
        # Convert the current target_pcd to a numpy array
        target_points = current_target_pcd
        
        # Concatenate the two point clouds
        merged_points = np.vstack((existing_points, target_points))
        
        # Update the global collision_pcd with the merged points
        collision_pcd = o3d.geometry.PointCloud()
        collision_pcd.points = o3d.utility.Vector3dVector(merged_points)
        
        # Update the visualization server
        lgr.info("Updated collision point cloud with the current target.")

    ''''start the main insert function'''
    xarm6_pk = XArm6WOEE()
    xarm = XArm6RealWorld(ip = arm_ip)
    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    ra = RobotArm(urdf_path= XARM6_WO_EE_URDF_PATH)
    env_pc = enviroment_constraint(xarm6_planner_cfg)
    xarm6_planner.mplib_add_point_cloud(env_pc, name="env_pc")
    xarm6_planner.mplib_add_point_cloud(np.asarray(collision_pcd.points), name = "collison_pcd")
    sv.scene.add_point_cloud("collision_pcd", points=np.asarray(collision_pcd.points), colors=(0, 0, 255), point_size=0.002, point_shape="circle")
    current_joint_values = np.array(xarm.get_joint_values())
    pre_align_pose, ee_target_pose = get_pre_pose_from_target_pose(arm_ip,target_pose)
    pcd_align_pose = target_pose.copy()
    pcd_align_pose[:3, 0] *= -1
    target_pcd = sample_points_from_mesh(mesh, 1000, seed=0)
    target_pcd, rotation_matrix, centroid = canonicalize_point_cloud_heu(target_pcd)
    homogenerous =  np.hstack((np.asarray(target_pcd.points), np.ones((np.asarray(target_pcd.points).shape[0],1))))
    transformed_points = (pcd_align_pose @ homogenerous.T).T
    transformed_points = transformed_points[:, :3] 
    sv.scene.add_point_cloud("target_pcd", points=transformed_points, colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    # print(f"pre_align_pose: {pre_align_pose}")
    ''' pick ten of the possible target pose, and sorted them according to the distance to the current pose'''
    # current_arm_mesh = ra.get_state_trimesh(current_joint_values, visual=True, collision=False)["visual"]
    # sv.scene.add_mesh_trimesh("current_arm_mesh", current_arm_mesh)
    status, ee_pose = xarm.get_position_se3()
    
    possible_target_pose = []
    for i in range(10):
        rotation = R.from_euler('x', 30 * i, degrees=True).as_matrix()
        target_pose[:3, :3] = target_pose[:3, :3] @ rotation
        pre_align_pose, ee_target_pose = get_pre_pose_from_target_pose(arm_ip, target_pose)
        possible_target_pose.append([pre_align_pose,ee_target_pose])
    print(f"shape of possible_target_pose: {np.array(possible_target_pose).shape}")
    possible_target_pose = sorted(possible_target_pose, key = lambda pose: se3_distance(pose[0], ee_pose))
    # print(f"possible target pose {possible_target_pose}")
    planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, possible_target_pose[0][0])
    # print(planning_result['status'])
    if planning_result['status']:
        pre_align_pose = possible_target_pose [0][0]
        ee_target_pose = possible_target_pose [0][1]
    # print(pre_align_pose)
    # print(ee_target_pose)
    
    '''try to change the rotation and plan again if is not success'''
    trial = 1
    while planning_result['status'] != 'Success' and trial < maximun_planning_time:
        lgr.info(f"Trial {trial}: Collision-free planning to pre-align position: Fail")
        planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, possible_target_pose[trial][0]) 
        if planning_result['status'] == 'Success':
            pre_align_pose = possible_target_pose[trial][0]
            ee_target_pose = possible_target_pose[trial][1]
            break     
        trial += 1

    if trial == maximun_planning_time:
        lgr.info("Collision-free planning to pre-align position: Fail")
        return
    # print(f"planing_result: {planning_result}")
    lgr.info("Collision-free planning to pre-align position: Success")
    
    '''visualize the pre-align pose'''
    sv.scene.add_frame("pre_align_pose", wxyz=R.from_matrix(pre_align_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pre_align_pose[:3, 3], axes_length=0.03, axes_radius=0.001)
    hand_open_mesh_path = Path("data/data_urdf/robot/xarm_gripper/hand_open.obj")
    gripper_trimesh = trimesh.load_mesh(hand_open_mesh_path).apply_scale(1.1)
    sv.scene.add_mesh_simple("hand_open", vertices=gripper_trimesh.vertices, faces=gripper_trimesh.faces, wxyz=R.from_matrix(pre_align_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pre_align_pose[:3, 3], opacity=0.5)
    waypt_joint_values_np = planning_result['position']
    end_joint_values = waypt_joint_values_np[-1]
    update_viser_mp_result(sv, xarm6_pk, current_joint_values, end_joint_values, waypt_joint_values_np)
    # validated = False
    # validate_button = sv.gui.add_button("Execute",)
    # # turn validated to True]
    # def validate_true():
    #     nonlocal validated
    #     validated = True
    #     lgr.info("validated")
    # validate_button.on_click(lambda _: validate_true())
    # while True:
    #     time.sleep(0.2)
    #     if validated:
    #         break
    # validated = False

    xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.1, wait=True)

    # bp()

    # # 规划运动到最终插入位置
    # current_joint_values = np.array(xarm.get_joint_values())

    # planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, ee_target_pose)
    # if planning_result['status'] != 'Success':
    #     lgr.info(f"Collision-free planning to target pose: Fail")
    #     xarm.arm.set_gripper_position(850, wait=True)
    #     home_joint_values = xarm.default_joint_values  # 默认的回到初始位置的关节角
    #     xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)
    #     return
    # lgr.info(f"Collision-free planning to target pose: Success")
    # waypt_joint_values_np = planning_result['position']

    # # 设置慢速插入
    # slow_speed = 0.01  # 插入速度
    # xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
    # xarm.set_joint_values(waypt_joint_values_np[-1], speed=slow_speed, wait=True)

    # 松开物体
    xarm.arm.set_gripper_position(330, wait=True)
    time.sleep(1)
    xarm.arm.set_gripper_position(850, wait=True)  # 松开夹爪
    # time.sleep(2)  # 确保物体释放完成

    # 当前末端执行器的位姿
    status, cur_ee_pose = xarm.get_position_se3()   # 获取末端执行器的当前位姿（4x4矩阵)
    # input("Press Enter to continue to lift the arm...")
    # 沿末端的 z 方向平移 10cm
    offset = np.array([0, 0, -0.28])  # 平移的偏移量
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = offset
    lifted_pose = cur_ee_pose @ translation_matrix  # 计算平移后的新位姿

    # 规划并执行移动
    status, lifted_joint_values = xarm6_planner.mplib_ik(np.array(xarm.get_joint_values()), lifted_pose)
    mp_is_success = status == 'Success'
    
    while not mp_is_success and offset[2] < -0.20:
        lgr.info(f"replanning z motiom")
        offset[2]+= 0.01
        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = offset
        lifted_pose = cur_ee_pose @ translation_matrix  # 计算平移后的新位姿
        status, lifted_joint_values = xarm6_planner.mplib_ik(np.array(xarm.get_joint_values()), lifted_pose)
        mp_is_success = status == 'Success'
        
    if not mp_is_success:
        lgr.info(f"Z planning: Fail")
        xarm.arm.set_gripper_position(850, wait=True)
        home_joint_values = xarm.default_joint_values
        xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)
    else:
        closest_lifted_arm_joint_value = get_closest_joint_value(current_joint_values, lifted_joint_values)
        lgr.info(f"z planning: Success")
        xarm.arm.set_servo_angle(angle=xarm.to_list(closest_lifted_arm_joint_value), speed=0.2, wait=True, is_radian=True)
#     time.sleep(3)

    # bp()
    
    # 返回初始位置
    home_joint_values = xarm.default_joint_values  # 默认的回到初始位置的关节角
    xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)

    lgr.info("Operation completed. Returned to home position.")
    update_collision_pcd(transformed_points)

                  
if __name__ == "__main__":

    initial = True  # only query GPT for assembly order in the first time.
    mask_list = []  # for order_planing
    bbox_list = []  # for order_planing
    collision_pcd = None
    target_list = []
    target_left = []
    target_right = []
    cam2leftbase = None
    cam2rightbase = None
    cam_K = None
    sv = viser.ViserServer()
    cam_serial = "241122074374"
    cam_K_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/K.npy")
    cam_K = np.load(cam_K_path)
    arm_cam_X_BaseCamera_path_r = Path(f"third_party/xarm6/data/camera/{cam_serial}/0107_excalib_capture00/optimized_X_BaseCamera.npy")
    cam2rightbase = np.load(arm_cam_X_BaseCamera_path_r)
    arm_cam_X_BaseCamera_path_l = Path(f"third_party/xarm6/data/camera/{cam_serial}/1219_excalib_capture00/optimized_X_BaseCamera.npy")
    cam2leftbase = np.load(arm_cam_X_BaseCamera_path_l)
    rotation_mat_path = '/home/shaol/data/zjx/rw/data/17_2/rotation_matrix.npy'
    center_path = '/home/shaol/data/zjx/rw/data/17_2/center.npy'
    target = 'original_part_01'
    mesh = trimesh.load(f"object_mesh_new/{target}/{target}.obj")


    
    pose_initialization(rotation_path=rotation_mat_path, center_path=center_path)
    target_left.sort(key = lambda x: x[1,3], reverse=True)
    target_right.sort(key = lambda x: x[1,3])
    pick(arm_ip=XARM6LEFT_IP)
    insert(arm_ip=XARM6LEFT_IP, target_pose=target_left[0], mesh = mesh)
    pick(arm_ip=XARM6_IP)
    insert(arm_ip=XARM6_IP, target_pose=target_right[0], mesh = mesh)
    pick(arm_ip=XARM6LEFT_IP)
    insert(arm_ip=XARM6LEFT_IP, target_pose=target_left[1], mesh = mesh)
    pick(arm_ip=XARM6_IP)
    insert(arm_ip=XARM6_IP, target_pose=target_right[1], mesh = mesh)

    # for i in range(2):
    #     # pick(arm_ip=XARM6LEFT_IP)
    #     insert(arm_ip=XARM6LEFT_IP)
    #     # pick(arm_ip=XARM6_IP)
    #     # insert(arm_ip=XARM6_IP)


    

        