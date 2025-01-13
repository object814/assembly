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
from utils import PointCloudUtils

def update_attach(xarm6_planner = None, object_name='sticker', pose_tool=None, pose_frank = None):

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

    trans_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) ** 2

    return trans_diff

def update_collision_pcd(current_target_pcd):
        """
        Updates the collision point cloud by merging the current target point cloud
        into the existing collision point cloud in the scene.
        
        Args:
            current_target_pcd: The point cloud of the current target object.
        """
        global collision_pcd  # Ensure we update the global collision_pcd
        if isinstance(current_target_pcd, o3d.geometry.PointCloud):
            current_target_pcd = np.asarray(current_target_pcd.points)
        elif not isinstance(current_target_pcd, np.ndarray):
            raise ValueError("Invalid type of point_cloud")
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
        pose = est.register(K=arm_cam_K, rgb=rgb, depth=depth, ob_mask=manual_mask_np, iteration=20)
        pose = arm_cam_X_BaseCamera @ pose

        points = sample_points_from_mesh(mesh, 50000, seed=0)

        object_pc_o3d = o3d.geometry.PointCloud()
        object_pc_o3d.points = o3d.utility.Vector3dVector(points)
        object_pc_o3d.transform(pose)

    return object_pc_o3d, pose

    

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
    if isinstance(point_cloud, o3d.geometry.PointCloud):
        point_cloud = np.asarray(point_cloud.points)
    elif isinstance(point_cloud, np.ndarray):
        point_cloud = point_cloud
    else:
        raise ValueError("Invalid type of point_cloud")
    
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
        base_rotation_cam  = rotation_mats[1]
        base_center_cam = centers[1]
        '''temple test!'''
        base_cam = np.eye(4)
        base_cam[:3, :3] = base_rotation_cam
        base_cam[:3, 3] = base_center_cam
        if len(centers)>2:
            canonical_base_pcd, base_mat, base_center = canonicalize_point_cloud_heu(base_pcd)
        else:
            # input('calculating with bbo')
            canonical_base_pcd, base_mat, base_center = PointCloudUtils.canonical_bbo(base_pcd, reverse_xy = True)
        sv.scene.add_point_cloud("canonical_pcd", points=canonical_base_pcd, colors=(0, 255, 0), point_size=0.002, point_shape="circle")
        canonical_base_mat = np.eye(4)
        canonical_base_mat[:3, :3] = base_mat.T
        canonical_base_mat[:3, 3] = base_center
        canonical_transform_base_cam = canonical_base_mat @ np.linalg.inv(base_cam) 

        sv.scene.add_frame("canonical_base_pose", wxyz=R.from_matrix(canonical_base_mat[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical_base_mat[:3, 3], axes_length=0.3, axes_radius=0.01)
        return canonical_transform_base_cam, rotation_mats, centers


def pick(object_name='box03', arm_ip=XARM6_IP):  

    def pick_policy_z_axis(pose_box, pcd_center, lengths):

        length = lengths[0]
        width = lengths[1]
        height = lengths[2]

        '''policy1 is to pick along the z axis of the object pose get from foundation psoe with 30cm pregrasp distance'''
        adjustment_rotation = R.from_euler('x', 90, degrees=True).as_matrix()
        reverse_rotation = R.from_euler('x' , -90, degrees= True).as_matrix()
        # Embed the adjustment rotation into a 4x4 transformation matrix
        rotation_z_towards_down = pose_box @ adjustment_rotation
        if rotation_z_towards_down[2,2]>0:
            rotation_z_towards_down = pose_box @ reverse_rotation

        # to see the origin pose

        # we need to grasp the top side of the board and its offset is half of width
        z_axis_offset = 0.5*width-0.03+0.17+0.1
        offset_in_object_frame = np.array([0, 0, -z_axis_offset]) 

        # 将偏移量从物体坐标系转换到世界坐标系
        offset_in_world_frame = rotation_z_towards_down @ offset_in_object_frame

        center_grasp = np.eye(4,4)
        center_grasp[:3, 3] = center_box+offset_in_world_frame
        center_grasp[:3, :3] = rotation_z_towards_down

        # center_grasp[2,3] += 0.3
        policy = "z_axis"

        print(f"center_grasp: {center_grasp}")
        return center_grasp, rotation_z_towards_down, policy
    
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
    print(f"home joint value is {home_joint_values}")
    xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)

    object_pc_o3d, X_WorldObject = get_object_pc_fp(object_name, arm_ip = arm_ip)
    center_box, pose_box, lengths = PointCloudUtils.extract_rectangle_with_pose(object_pc_o3d)
    print(f"lengths is {lengths}")
    pose_box = np.array(pose_box, copy=True)
    print(f"center_box: {center_box}, pose_box: {pose_box}")
    pcd_center = get_pcd_center(object_pc_o3d)
    sv.scene.add_point_cloud("original_pc", points=np.asarray(object_pc_o3d.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    # sv.scene.add_point_cloud("canonical_pc", points=np.asarray(canonicalized_pcd.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    sv.scene.add_frame("canonical_pose", wxyz=R.from_matrix(pose_box).as_quat()[[3, 0, 1, 2]], position=pcd_center, axes_length=0.03, axes_radius=0.001)
    grasp_pose, adjusted_rotation_matrix, policy = pick_policy_z_axis(pose_box, pcd_center, lengths)
    wxyz = R.from_matrix(adjusted_rotation_matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]  # 转换为 wxyz 格式
    sv.scene.add_frame("z_down_pose", wxyz=wxyz, position=center_box, axes_length=0.03, axes_radius=0.001)
    sv.scene.add_point_cloud("object_pc", points=np.asarray(object_pc_o3d.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
    current_joint_values = np.array(xarm.get_joint_values())

    # REAL PALNNING IS HERE! 
    planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, grasp_pose)
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
    xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.35, wait=True)

    # input("Press Enter to continue...")
    
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    current_joint_values = np.array(xarm.get_joint_values()) # yiwen

    ''' grasping motion is here'''
    if arm_ip==XARM6_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.1]) 
    elif arm_ip==XARM6LEFT_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.1]) # 物体局部坐标系下向下偏移
    if policy == "top_down":
        center_grasp = grasp_pose.copy()
        center_grasp[:3, 3] = grasp_pose[:3,3]- offset_grasp_in_object_frame
        print(f"center_grasp: {center_grasp}")
    elif policy=="z_axis":
        center_grasp = grasp_pose.copy()
        offset_grasp_in_world_frame = adjusted_rotation_matrix @ offset_grasp_in_object_frame
        center_grasp[:3, 3] = grasp_pose[:3,3] + offset_grasp_in_world_frame

        print(f"center_grasp: {center_grasp}")
    
    
    status, grasp_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, center_grasp)
    print(f"status: {status}")
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
        
        if not mp_is_success:
            lgr.info(f"Lift planning: Fail")
        else:
            lgr.info(f"Lift planning: Success")
            closest_lifted_arm_joint_value = get_closest_joint_value(current_joint_values, lifted_arm_joint_values)
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
        print(f"ee pose is {ee_pose}, frank pose is {frank_pose}")
        update_attach(object_name=object_name, pose_tool=ee_pose,pose_frank = frank_pose)
        # bp()           

def get_target_pose(arm_ip = XARM6LEFT_IP, rotation_path = None , center_path = None):

    global collision_pcd
    '''here we get all the target pose based on the left arm base'''
    # sv = viser.ViserServer()
    base_pcd, pose = get_object_pc_fp(object_name='box02',arm_ip=arm_ip)
    collision_pcd = base_pcd
    update_collision_pcd(collision_pcd)
    sv.scene.add_point_cloud("base_pcd", points = np.asarray(base_pcd.points), colors = (0,255,0),point_size = 0.01, point_shape = 'circle')

    '''' get taget pose based on the canonical pose and assume the base in the real-world is static'''
    trans_mat, rotation_mats, centers= calculate_transform_matrix(rotation_path, center_path, base_pcd)
    
    # for i in range(1,len(rotation_mats)):
    target_pose = np.eye(4)
    # in the box case, the rotation definition is little bit different from the stool
    target_pose[:3, :3] = rotation_mats[0]
    target_pose[:3, 3] = centers[0]
    target_pose = trans_mat @ target_pose
    target_list.append(target_pose)
        # sv.scene.add_frame(f"target_pose{i}", wxyz=R.from_matrix(target_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=target_pose[:3, 3], axes_length=0.3, axes_radius=0.01)
    sorted_list = sorted(target_list, key = lambda x: x[0,3])
    global target_left, target_right
    if len(target_list) > 1 and len(target_list)%2==0 :
        target_left = sorted_list[:len(sorted_list)//2]
        target_right = sorted_list[len(sorted_list)//2:]
    else:
        target_left = sorted_list
        target_right = sorted_list
    # print(f"cam2leftbase: {cam2leftbase}")
    # print(f"cam2rightbase: {cam2rightbase}")
    mat_left_to_right = cam2rightbase@ np.linalg.inv(cam2leftbase)
    # print(f"mat_left_to_right: {mat_left_to_right}")
    target_right = [mat_left_to_right @ mat for mat in target_right]
    # print(f"target_right: {target_right}")

def get_ee_target_from_initial(object_name = 'box01', target_pose = None):
    initial_pcd, initial_pose = get_object_pc_fp(object_name='box01', arm_ip=XARM6_IP)
    cano_pcd, rotation_mat, center = PointCloudUtils.canonical_bbo(initial_pcd, visualize = True)
    initial_cano_pose = np.eye(4)
    initial_cano_pose[:3,:3] = rotation_mat.T
    initial_cano_pose[:3,3] = center
    transfer_mat = target_right[0] @ np.linalg.inv(initial_cano_pose)
    bp()
    xarm = XArm6RealWorld(ip = XARM6_IP)
    xarm6_pk = XArm6WOEE()
    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    _, xarm_frank = xarm.get_position()
    bp()
    ee_target_pose = transfer_mat @ xarm_frank
    bp()


    pass
def base_to_target_pose():

    pass

                  
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
    rotation_mat_path = '/home/shaol/data/zjx/rw/data/box111/rotation_matrix.npy'
    center_path = '/home/shaol/data/zjx/rw/data/box111/center.npy'
    target = 'original_part_01'
    mesh = trimesh.load(f"object_mesh_new/{target}/{target}.obj")

    # sv.scene.add_frame("target_right", wxyz=R.from_matrix(target_right[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_right[:3,3], axes_length=0.3, axes_radius=0.01)
    # sv.scene.add_frame("initial_ee_pose", wxyz=R.from_matrix(initial_ee_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=initial_ee_pose[:3,3], axes_length=0.3, axes_radius=0.01)
    # # sv.scene.add_frame("initial_frank_pose", wxyz=R.from_matrix(initial_frank_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=initial_frank_pose[:3,3], axes_length=0.3, axes_radius=0.01)
    # bp()
    # pose_initialization(rotation_path=rotation_mat_path, center_path=center_path)
    # target_left.sort(key = lambda x: x[1,3], reverse=True)
    # target_right.sort(key = lambda x: x[1,3])
    pick(arm_ip=XARM6_IP, object_name='box01')
    pick(arm_ip=XARM6LEFT_IP, object_name='box02')
    xarm = XArm6RealWorld(ip = XARM6LEFT_IP)
    first_base_joints = np.array([ 0.03665191, -0.93724181, -0.44331363,  0.12042772,  0.5969026 ,  1.57603231 ])
    xarm.set_joint_values(first_base_joints, speed=0.35, wait=True)
    get_target_pose(arm_ip=XARM6LEFT_IP, rotation_path = rotation_mat_path, center_path= center_path)
    print(f"target left is{target_left}, target right is {target_right}")
    target_right = target_right[0]
    target_left = target_left[0]
    sv.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    bp()
    '''add a viser for right arm'''
    sv2 = viser.ViserServer()
    initial_pcd, initial_pose = get_object_pc_fp(object_name='box01', arm_ip=XARM6_IP)
    cano_pcd, rotation_mat, center = PointCloudUtils.canonical_bbo(initial_pcd, visualize = False, reverse_xy = True)
    sv2.scene.add_point_cloud("cano_pcd", points = cano_pcd, colors = (0,255,0),point_size = 0.01, point_shape = 'circle')
    sv2.scene.add_point_cloud("ini_pcd", points = np.asarray(initial_pcd.points), colors = (0,255,0),point_size = 0.01, point_shape = 'circle')
    
    # bp()
    initial_cano_pose = np.eye(4)
    initial_cano_pose[:3,:3] = rotation_mat.T
    initial_cano_pose[:3,3] = center

    # sv.scene.add_frame("canonical_pose", wxyz=R.from_matrix(initial_cano_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=initial_cano_pose[:3,3], axes_length=0.3, axes_radius=0.01)
    # bp()
    transfer_mat = target_right @ np.linalg.inv(initial_cano_pose)
    bp()
    xarm = XArm6RealWorld(ip = XARM6_IP)
    xarm6_pk = XArm6WOEE()
    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    _, xarm_ee = xarm.get_position()
    sv2.scene.add_frame("xarm_ee", wxyz=R.from_matrix(xarm_ee[:3,:3]).as_quat()[[3, 0, 1, 2]], position=xarm_ee[:3,3], axes_length=0.3, axes_radius=0.01)
    bp()
    ee_target_pose = transfer_mat @ xarm_ee
    sv2.scene.add_frame("frank_target_pose", wxyz=R.from_matrix(ee_target_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=ee_target_pose[:3,3], axes_length=0.3, axes_radius=0.01)
    bp()

    status, ee_pose = xarm.get_position_se3()
    status2, frank_pose = xarm.get_position()
    print(f"ee pose is {ee_pose}, frank pose is {frank_pose}")
    update_attach(object_name='box01', pose_tool=ee_pose,pose_frank = frank_pose)
    
    planning_result = xarm6_planner.mplib_plan_pose(np.array(xarm.get_joint_values()), ee_target_pose) 
    
    waypt_joint_values_np = planning_result['position']
    end_joint_values = waypt_joint_values_np[-1]
    xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.1, wait=True)
    # second_base_joints = np.array([ 0.04712389, -0.27925268, -1.37357412,  0.06283185,  1.57603231,  1.57603231 ])
    # xarm.set_joint_values(second_base_joints, speed=0.35, wait=True)
    # insert(arm_ip=XARM6LEFT_IP, target_pose=target_left[0], mesh = mesh)
    # pick(arm_ip=XARM6_IP)
    # insert(arm_ip=XARM6_IP, target_pose=target_right[0], mesh = mesh)
    # pick(arm_ip=XARM6LEFT_IP)
    # insert(arm_ip=XARM6LEFT_IP, target_pose=target_left[1], mesh = mesh)
    # pick(arm_ip=XARM6_IP)
    # insert(arm_ip=XARM6_IP, target_pose=target_right[1], mesh = mesh)

    # for i in range(2):
    #     # pick(arm_ip=XARM6LEFT_IP)
    #     insert(arm_ip=XARM6LEFT_IP)
    #     # pick(arm_ip=XARM6_IP)
    #     # insert(arm_ip=XARM6_IP)


    

        