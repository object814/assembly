import cv2
import time
import viser 
import numpy as np
import open3d as o3d
import sys
import os
# import hydra
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR+"/3rdparty/segment-anything"))
sys.path.append(os.path.join(ROOT_DIR+"/3rdparty/xarm6"))
import torch
import warnings

from xarm6_interface import XARM6_IP, XARM6LEFT_IP
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


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

X_BaserightBaseleft = np.array([
        [-0.9994789, -0.02021456, -0.02516249, 1.1056788],
        [0.02050998, -0.99972314, -0.01153819, -0.03005717],
        [-0.02492228, -0.01204827, 0.9996168, 0.00512242],
        [0.0, 0.0, 0.0, 1.0]
    ])


mat_for_stick = np.array([[-0.8072, -0.1237, -0.5772, -0.3197],
                          [0.1362, 0.9124, -0.3860, -0.2194],
                          [0.5744, -0.3902, -0.7196, -1.1432],
                          [0.0, 0.0, 0.0, 1.0]])

mat_for_board = np.array([[0.7562, 0.4688, 0.4564, -0.0353],
                          [0.4688, 0.8638, -0.4158,  -0.0648],
                          [0.4564, 0.1846, 0.7866, -1.2246],
                          [0.0, 0.0, 0.0, 1.0]])

trans_cam_world = np.zeros((4, 4))
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
    # 4. 获取旋转矩阵（主轴方向）
    rotation_matrix = pca.components_
    print(f"rotation_matrix: {rotation_matrix}")
    transformed_pcd = np.dot(centered_pcd, rotation_matrix.T)
    # restored_pcd = centered_pcd + centroid

    # 将还原的点云转换回 Open3D 点云对象
    canonicalized_pcd = o3d.geometry.PointCloud()
    canonicalized_pcd.points = o3d.utility.Vector3dVector(transformed_pcd)
    return canonicalized_pcd, rotation_matrix, centroid

def get_object_pc_fp(object_name, arm_ip=XARM6_IP):
    object_name_dino = object_name.replace("_", " ") + "."
    # Initialize the SAM predictor
    # sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_PATH)
    # sam.to(device="cuda")
    # sam_predictor = SamPredictor(sam)
    mesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj")
    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()
    est = FoundationPose(
        model_pts=mesh.vertices, 
        model_normals=mesh.vertex_normals, 
        mesh=mesh, 
        scorer=scorer, 
        refiner=refiner,
        glctx=glctx
    )
    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    cam_serial = "241122074374"
    # top_cam_serial = '233622079809'
    camera_serial_nums = [cam_serial]
    multi_rs = MultiRealsense(camera_serial_nums)

    if arm_ip == XARM6LEFT_IP:
        arm_left_cam_K_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/K.npy")
        arm_left_cam_K = np.load(arm_left_cam_K_path)
        # arm_right_cam_X_BaseCamera_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/1230_excalib_capture00/optimized_X_BaseCamera.npy")
        # arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)
        # arm_left_cam_X_BaseCamera = np.linalg.inv(X_BaserightBaseleft)@arm_right_cam_X_BaseCamera
        arm_left_cam_X_BaseCamera_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/1219_excalib_capture00/optimized_X_BaseCamera.npy")
        arm_left_cam_X_BaseCamera = np.load(arm_left_cam_X_BaseCamera_path)
        multi_rs.set_intrinsics(0, arm_left_cam_K[0, 0], arm_left_cam_K[1, 1], arm_left_cam_K[0, 2], arm_left_cam_K[1, 2])
        camera_wxyzs = [
            R.from_matrix(arm_left_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
        ]
        camera_positions = [arm_left_cam_X_BaseCamera[:3, 3]]
        X_BaseCamera_list = [arm_left_cam_X_BaseCamera]
        
        for i in range(50):
            multi_rs.getCurrentData()
        rtr_dict_list = multi_rs.getCurrentData()
        
        for camera_idx in range(len(rtr_dict_list)):
            rtr_dict = rtr_dict_list[camera_idx]
            
            rgb = rtr_dict["rgb"]
            depth = (rtr_dict["depth"].astype(np.float32) / 1000).astype(np.float32)
            pc_o3d = rtr_dict["pointcloud_o3d"]
            
            prompt_drawer.reset()
            mask_np = prompt_drawer.run(rgb)  # (720, 1280)
            
            
            pose = est.register(K=arm_left_cam_K, rgb=rgb, depth=depth, ob_mask=mask_np, iteration=10)
            # from cam to the world cordination
            pose = arm_left_cam_X_BaseCamera @ pose
            
            points, face_indices = mesh.sample(1024, return_index=True)
            normals = mesh.face_normals[face_indices]
            
            object_pc_o3d = o3d.geometry.PointCloud()
            object_pc_o3d.points = o3d.utility.Vector3dVector(points)
            object_pc_o3d.normals = o3d.utility.Vector3dVector(normals)
            object_pc_o3d.transform(pose)
        
    elif arm_ip == XARM6_IP:
        arm_right_cam_K_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/K.npy")
        arm_right_cam_K = np.load(arm_right_cam_K_path)
        arm_right_cam_X_BaseCamera_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/1230_excalib_capture00/optimized_X_BaseCamera.npy")
        arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)
        multi_rs.set_intrinsics(0, arm_right_cam_K[0, 0], arm_right_cam_K[1, 1], arm_right_cam_K[0, 2], arm_right_cam_K[1, 2])
        
        for i in range(50):
            multi_rs.getCurrentData()
        rtr_dict_list = multi_rs.getCurrentData()
        
        for camera_idx in range(len(rtr_dict_list)):
            rtr_dict = rtr_dict_list[camera_idx]
            
            rgb = rtr_dict["rgb"]
            depth = (rtr_dict["depth"].astype(np.float32) / 1000).astype(np.float32)
            pc_o3d = rtr_dict["pointcloud_o3d"]
            
            prompt_drawer.reset()
            mask_np = prompt_drawer.run(rgb)  # (720, 1280)
            
            
            pose = est.register(K=arm_right_cam_K, rgb=rgb, depth=depth, ob_mask=mask_np, iteration=20)
            # from cam to the world cordination
            pose = arm_right_cam_X_BaseCamera @ pose
            
            points, face_indices = mesh.sample(1024, return_index=True)
            normals = mesh.face_normals[face_indices]
            
            object_pc_o3d = o3d.geometry.PointCloud()
            object_pc_o3d.points = o3d.utility.Vector3dVector(points)
            object_pc_o3d.normals = o3d.utility.Vector3dVector(normals)
            object_pc_o3d.transform(pose)

    else: raise ValueError("Invalid arm_ip")
            
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
    
def filter_top_down_grasps(X_WorldEE, clip_min_z=0.05, approach_direction="z"):
    '''
    X_WorldEE: shape=(n, 4, 4)
    '''
    alphabet_to_idx = {"x": 0, "y": 1, "z": 2}
    axis_idx = alphabet_to_idx[approach_direction]
    
    # filter the invalid z
    valid_z_mask = X_WorldEE[:, 2, 3] > clip_min_z
    
    # only keep the top down grasps: get the approach direction's 3rd element
    valid_top_down_mask = X_WorldEE[:, 2, axis_idx] < -0.3
    valid_side_mask = X_WorldEE[:, 0, axis_idx] > -0.3

    # combine the two masks
    valid_mask = valid_z_mask & valid_top_down_mask & valid_side_mask
    
    # return the valid grasps
    return X_WorldEE[valid_mask]




planner_timestep = 1.0 / 50.0
cmd_timestep = 1.0 / 100.0 
pregrasp_retreat_distance = 0.08
# @hydra.main(version_base="1.2", config_path="", config_name="validate")



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
    
    
    
 
    ''' setup the planner and vis '''
    # sv = viser.ViserServer()
    xarm6_pk = XArm6WOEE()
    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    env_params = WoodenTableMount()
    workspace_pc = create_bounding_box_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    table_plane_pc = create_plane_pc(env_params.table_plane_xmin, env_params.table_plane_ymin, env_params.table_plane_zmin, env_params.table_plane_xmax, env_params.table_plane_ymax, env_params.table_plane_zmax, xarm6_planner_cfg.n_env_pc)
    workspace_xmin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmin, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax, env_params.ymin, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymax_pc = create_plane_pc(env_params.xmin, env_params.ymax, env_params.zmin, env_params.xmax, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    env_pc = np.concatenate([workspace_pc, table_plane_pc, workspace_xmin_pc, workspace_ymin_pc, workspace_ymax_pc], axis=0)
    env_pc = env_pc_post_process(env_pc, filter_norm_thresh=0.1, n_save_pc=None)
    xarm6_planner.mplib_add_point_cloud(env_pc, name="env_pc")
    ''' setup the planner and vis '''

    xarm = XArm6RealWorld(ip = arm_ip)
    xarm.arm.set_gripper_position(850, wait=True)
    home_joint_values = xarm.default_joint_values  # 默认的回到初始位置的关节角
    xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)
    ''' get the object pc and pose'''
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
    
    input("Press Enter to continue...")
    
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    current_joint_values = np.array(xarm.get_joint_values()) # yiwen

    ''' grasping motion is here'''
    if arm_ip==XARM6_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.14]) 
    elif arm_ip==XARM6LEFT_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.13]) # 物体局部坐标系下向下偏移13cm  
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
            time.sleep(2)
            # xarm.arm.set_gripper_position(850, wait=True)

            input("Press Enter to continue...")
            
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

    def get_current_attach_pose_offset(object_name='sticker'):
        attach_pcd, attch_pose = get_object_pc_fp(object_name,arm_ip)
        attach_pcd_center = get_pcd_center(attach_pcd)
        status, position = xarm.get_position()
        print(f"current position: {position}")
        print(f"attach_pcd_center: {attach_pcd_center}")

        if status != 0:
            lgr.error("Failed to get current position")
            return
        grasp_offset = 0.001*np.array(position[:3]) - np.array(attach_pcd_center)
        print(f"grasp_offset: {grasp_offset}")
        
        return grasp_offset
    
    def update_attach(object_name='sticker', pose=None):
        grippermount_data_dir = Path("data/data_urdf/robot/xarm_gripper/hand_open_cvx_hull.obj")
        gripper_trimesh = trimesh.load_mesh(grippermount_data_dir).apply_scale(1.1)
        gripper_transform = np.eye(4)
        xarm6_planner.mplib_update_attached_object(
            gripper_trimesh,
            gripper_transform,
        )
        
        ####zjx, attach trimesh
        attached_trimesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj").apply_scale(1.1)
        # attached_trimesh_cvx_hull = attached_trimesh.convex_hull
        # Create a transformation matrix for attaching (identity for testing)
        
        attach_transform = pose  # 4x4 transformation matrix, identity for testing

        # Get position and rotation from the transformation matrix
        position = attach_transform[:3, 3]
        rotation_matrix = attach_transform[:3, :3]
        wxyz = R.from_matrix(rotation_matrix).as_quat()[[3, 0, 1, 2]]  # Convert to wxyz format

        # Update the planner with the attached object
        xarm6_planner.mplib_update_attached_object(attached_trimesh, attach_transform)

        # Visualize the attached object in the scene
        sv.scene.add_mesh_simple(
            "attached_object",
            vertices=attached_trimesh.vertices,
            faces=attached_trimesh.faces,
            wxyz=wxyz,
            position=position,
            opacity=0.5,
        )
        #####################################
            



def insert(arm_ip=XARM6_IP):
    global initial, collision_pcd  # 声明使用全局变量
    maximun_planning_time = 10

    def get_pre_pose_from_target_pose(target_pose, ee_offset=0.166):
        ee_offset = ee_offset  # 末端执行器距离法兰的距离
        ee_offset_vector = np.array([0, 0, -ee_offset])  # 在目标坐标系下的偏移向量

        ee_target_pose = target_pose.copy()
        # 应用偏移
        ee_target_pose[:3, 3] += target_pose[:3, :3] @ ee_offset_vector.reshape(3)
        # ee_target_pose[:3, 3] -= ee_offset_grasp
        # 沿目标姿态的 x 轴偏移 10 cm
        x_axis_offset = -0.1  # 10 cm
        pre_align_pose = ee_target_pose.copy()
        x_offset_vector = np.array([x_axis_offset, 0, 0])  # 在目标坐标系下的 x 方向偏移向量
        pre_align_pose[:3, 3] += target_pose[:3, :3] @ x_offset_vector  # 应用偏移

        return pre_align_pose, ee_target_pose

    def get_current_attach_pose_offset(object_name='sticker'):
        attach_pcd, attch_pose = get_object_pc_fp(object_name,arm_ip)
        attach_pcd_center = get_pcd_center(attach_pcd)
        status, position = xarm.get_position()
        print(f"current position: {position}")
        print(f"attach_pcd_center: {attach_pcd_center}")

        if status != 0:
            lgr.error("Failed to get current position")
            return
        grasp_offset = 0.001*np.array(position[:3]) - np.array(attach_pcd_center)
        print(f"grasp_offset: {grasp_offset}")  
        return grasp_offset
    
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
        target_points = np.asarray(current_target_pcd.points)
        
        # Concatenate the two point clouds
        merged_points = np.vstack((existing_points, target_points))
        
        # Update the global collision_pcd with the merged points
        collision_pcd = o3d.geometry.PointCloud()
        collision_pcd.points = o3d.utility.Vector3dVector(merged_points)
        
        # Update the visualization server
        lgr.info("Updated collision point cloud with the current target.")

    
    xarm6_pk = XArm6WOEE()
    xarm = XArm6RealWorld(ip = arm_ip)
    # sv = viser.ViserServer()
    if  initial == True:
        collision_pcd, pose = get_object_pc_fp(object_name='banzi',arm_ip=arm_ip)
        initial = False
    sv.scene.add_point_cloud("obstacle_pc", points=np.asarray(collision_pcd.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
    
    '''' get a taget pose until is correct'''
    correct_target_pose = False
    while not correct_target_pose:
        target_pcd,target_pose = get_object_pc_fp(object_name='sticker',arm_ip=arm_ip)
        x_axis = target_pose[:3, 0]
        z_x_axis = x_axis[2]
        if z_x_axis > 0:
            lgr.info("The target pose is incorrect, please adjust the pose.")
            continue
        else:
            lgr.info("The target pose is correct, please continue.")
            break

    target_center = get_pcd_center(target_pcd)
    target_pose[:3, 3] = target_center
    print(f"target pose: {target_pose}")
    sv.scene.add_point_cloud("target", points=np.asarray(target_pcd.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")

    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    xarm6_planner.mplib_add_point_cloud(np.asarray(collision_pcd.points), name="collision_pc")

    current_joint_values = np.array(xarm.get_joint_values())
    pre_align_pose, ee_target_pose = get_pre_pose_from_target_pose(target_pose)
    print(f"pre_align_pose: {pre_align_pose}")
    planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, pre_align_pose)
    '''try to change the rotation and plan again if is not success'''
    trial = 0

    while planning_result['status'] != 'Success' and trial < maximun_planning_time:
        lgr.info(f"Trial {trial}: Collision-free planning to pre-align position: Fail")
        
        # 更新旋转矩阵
        rotation = R.from_euler('x', 30 * trial, degrees=True).as_matrix()
        target_pose[:3, :3] = target_pose[:3, :3] @ rotation
        pre_align_pose, ee_target_pose = get_pre_pose_from_target_pose(target_pose)

        # 尝试规划
        planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, pre_align_pose)
        trial += 1

    if trial == maximun_planning_time:
        lgr.info("Collision-free planning to pre-align position: Fail")
        return

    lgr.info("Collision-free planning to pre-align position: Success")
    
    '''visualize the pre-align pose'''
    sv.scene.add_frame("pre_align_pose", wxyz=R.from_matrix(pre_align_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pre_align_pose[:3, 3], axes_length=0.03, axes_radius=0.001)
    hand_open_mesh_path = Path("data/data_urdf/robot/xarm_gripper/hand_open.obj")
    gripper_trimesh = trimesh.load_mesh(hand_open_mesh_path).apply_scale(1.1)
    sv.scene.add_mesh_simple("hand_open", vertices=gripper_trimesh.vertices, faces=gripper_trimesh.faces, wxyz=R.from_matrix(pre_align_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pre_align_pose[:3, 3], opacity=0.5)
    waypt_joint_values_np = planning_result['position']
    end_joint_values = waypt_joint_values_np[-1]
    update_viser_mp_result(sv, xarm6_pk, current_joint_values, end_joint_values, waypt_joint_values_np)
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
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.1, wait=True)

    input("Press Enter to continue to final insertion pose...")

    # 规划运动到最终插入位置
    current_joint_values = np.array(xarm.get_joint_values())

    planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, ee_target_pose)
    if planning_result['status'] != 'Success':
        lgr.info(f"Collision-free planning to target pose: Fail")
        return
    lgr.info(f"Collision-free planning to target pose: Success")
    waypt_joint_values_np = planning_result['position']

    # 设置慢速插入
    slow_speed = 0.01  # 插入速度
    xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=slow_speed, wait=True)

    # 松开物体
    xarm.arm.set_gripper_position(850, wait=True)  # 松开夹爪
    time.sleep(2)  # 确保物体释放完成

    # 当前末端执行器的位姿
    current_pose = target_pose  # 获取末端执行器的当前位姿（4x4矩阵）
    input("Press Enter to continue to lift the arm...")
    # 沿末端的 z 方向平移 10cm
    offset = np.array([0, 0, -0.3])  # 平移的偏移量
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = offset
    lifted_pose = current_pose @ translation_matrix  # 计算平移后的新位姿

    # 规划并执行移动
    status, lifted_joint_values = xarm6_planner.mplib_ik(np.array(xarm.get_joint_values()), lifted_pose)
    mp_is_success = status == 'Success'
    
    if not mp_is_success:
        lgr.info(f"z planning: Fail")
        return
    else:
        closest_lifted_arm_joint_value = get_closest_joint_value(current_joint_values, lifted_joint_values)
        lgr.info(f"z planning: Success")
        xarm.arm.set_servo_angle(angle=xarm.to_list(closest_lifted_arm_joint_value), speed=0.2, wait=True, is_radian=True)
        time.sleep(3)

    # 返回初始位置
    home_joint_values = xarm.default_joint_values  # 默认的回到初始位置的关节角
    xarm.set_joint_values(home_joint_values, speed=0.35, wait=True)

    lgr.info("Operation completed. Returned to home position.")

    update_collision_pcd(target_pcd)


                  
if __name__ == "__main__":
    # from hydra.core.global_hydra import GlobalHydra
    # GlobalHydra.instance().clear()  # 清除全局状态
    # Global variables
    initial = True
    collision_pcd = None
    sv = viser.ViserServer()
    
    for i in range(2):
        pick(arm_ip=XARM6LEFT_IP)
        insert(arm_ip=XARM6LEFT_IP)
        pick(arm_ip=XARM6_IP)
        insert(arm_ip=XARM6_IP)


    

        