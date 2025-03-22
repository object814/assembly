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

def se3_distance(pose1, pose2):

    trans_diff = np.linalg.norm(pose1[:3, 3] - pose2[:3, 3]) ** 2

    return trans_diff

def enviroment_constraint(xarm6_planner_cfg):
    env_params = WoodenTableMount()
    workspace_pc = create_bounding_box_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    # table_plane_pc = create_plane_pc(env_params.table_plane_xmin, env_params.table_plane_ymin, env_params.table_plane_zmin, env_params.table_plane_xmax, env_params.table_plane_ymax, env_params.table_plane_zmax, xarm6_planner_cfg.n_env_pc)
    workspace_xmin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmin, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax, env_params.ymin, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymax_pc = create_plane_pc(env_params.xmin, env_params.ymax, env_params.zmin, env_params.xmax, env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    env_pc = np.concatenate([workspace_pc, workspace_xmin_pc, workspace_ymin_pc, workspace_ymax_pc], axis=0)
    env_pc = env_pc_post_process(env_pc, filter_norm_thresh=0.1, n_save_pc=None)
    return env_pc

def sample_points_from_mesh(mesh, num_points, seed=None):
    if seed is not None:
        np.random.seed(seed)  # 设置随机数种子
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points


def get_object_pc_fp(object_name, arm_ip=XARM6_IP):

    global cam2leftbase, cam2rightbase, cam_serial, cam_K
    global initial, mask_list, bbox_list  # for order_planing
    object_name_dino = object_name.replace("_", " ") + "."

    mesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj")
    scorer, refiner, glctx = ScorePredictor(), PoseRefinePredictor(), dr.RasterizeCudaContext()
    est = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=scorer, refiner=refiner, glctx=glctx)
    lgr.info(f"Please draw box for {object_name}")
    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    arm_cam_K = cam_K
    if arm_ip == XARM6LEFT_IP:
        arm_cam_X_BaseCamera = cam2leftbase
        sv = sv_left
    elif arm_ip == XARM6_IP:
        arm_cam_X_BaseCamera = cam2rightbase 
        sv = sv_right
    else:
        raise ValueError("Invalid arm_ip")    
    multi_rs = MultiRealsense([cam_serial])
    multi_rs.set_intrinsics(0, arm_cam_K[0, 0], arm_cam_K[1, 1], arm_cam_K[0, 2], arm_cam_K[1, 2])
    
    while True:
        for _ in range(50):
            multi_rs.getCurrentData()
            rtr_dict_list = multi_rs.getCurrentData()
        for rtr_dict in rtr_dict_list:
            rgb, depth = rtr_dict["rgb"], (rtr_dict["depth"].astype(np.float32) / 1000).astype(np.float32)
            prompt_drawer.reset()
            manual_mask_np = prompt_drawer.run(rgb)

            pose = est.register(K=arm_cam_K, rgb=rgb, depth=depth, ob_mask=manual_mask_np, iteration=20)
            pose = arm_cam_X_BaseCamera @ pose

            points = sample_points_from_mesh(mesh, 10000, seed=0)
            pcd_center = np.mean(points, axis=0)
            object_pc_o3d = o3d.geometry.PointCloud()
            object_pc_o3d.points = o3d.utility.Vector3dVector(points)
            object_pc_o3d.transform(pose)

            sv.scene.add_point_cloud("object_pcd", points=np.asarray(object_pc_o3d.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
            sv.scene.add_frame("object_pose", wxyz=R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pcd_center, axes_length=0.03, axes_radius=0.001)


            lgr.info("Please validate the plan in the Viser GUI.")
            
            validated = False
            retry = False

            validate_button = sv.gui.add_button("Validated")
            retry_button = sv.gui.add_button("Retry")

            def validate_true():
                nonlocal validated
                validated = True
                lgr.info("Validated button clicked.")

            def retry_true():
                nonlocal retry
                retry = True
                lgr.info("Retry button clicked.")

            validate_button.on_click(lambda _: validate_true())
            retry_button.on_click(lambda _: retry_true())

            while not (validated or retry):
                time.sleep(0.2)

            if validated:
                return object_pc_o3d, pose, pcd_center

            if retry:
                lgr.info("Retrying...")
                break

    

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




planner_timestep = 1.0 / 50.0
cmd_timestep = 1.0 / 100.0 
pregrasp_retreat_distance = 0.08
# @hydra.main(version_base="1.2", config_path="", config_name="validate")

def pick(object_name='box03', xarm = None, policy = None):  
    '''init the arm in home pose'''
    xarm = xarm
    xarm.move_gripper(open = True, wait = True)
    xarm.move_to_home()
    '''get the pose by foundation pose'''
    object_pc_o3d, X_WorldObject, pcd_center = get_object_pc_fp(object_name, arm_ip = xarm.ip)
    sv_right.scene.add_frame("pose_from_fdp", wxyz=R.from_matrix(X_WorldObject[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pcd_center, axes_length=0.03, axes_radius=0.001)

    center_box, pose_box, lengths = PointCloudUtils.extract_rectangle_with_pose(object_pc_o3d)
    print(f"width is {lengths[1]}")
    pose_box = np.array(pose_box, copy=True)
    pcd_center = get_pcd_center(object_pc_o3d)
    # sv.scene.add_point_cloud("canonical_pc", points=np.asarray(canonicalized_pcd.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    if policy == 'shelf': 
        grasp_pose, adjusted_rotation_matrix, policy = Grasp_Policy.pick_policy_for_shelf(X_WorldObject[:3,:3], pcd_center, lengths[1])
    elif policy == 'box':
        grasp_pose, adjusted_rotation_matrix, policy = Grasp_Policy.pick_policy_z_axis(pose_box, pcd_center, lengths)
    if xarm.ip==XARM6_IP:
        sv_right.scene.add_frame("grasp_pose", wxyz=R.from_matrix(grasp_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=grasp_pose[:3,3], axes_length=0.03, axes_radius=0.001)
        # sv_right.scene.add_point_cloud("object_in_world", points=np.asarray(object_pc_o3d.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    else:
        sv_left.scene.add_frame("grasp_pose", wxyz=R.from_matrix(grasp_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=grasp_pose[:3,3], axes_length=0.03, axes_radius=0.001)
        # sv_left.scene.add_point_cloud("object_in_world", points=np.asarray(object_pc_o3d.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    if policy == "shelf":
        center_grasp = grasp_pose.copy()
        # if center_grasp[2, 2]> 0:
        #     center_grasp[:3, :3] = center_grasp[:3, :3] @ R.from_euler('y', 180, degrees=True).as_matrix()
        X_WorldEeflift =  center_grasp.copy()
        X_WorldEeflift[:3, 3] += adjusted_rotation_matrix @ np.array([0, 0, -0.15])
        xarm.plan_and_execute(X_WorldEeflift)
        xarm.move_gripper(close =True, wait=True)
        xarm.move_to_home()
        return
    '''move to grasp pose'''
    xarm.plan_and_execute(grasp_pose)

    ''' grasping motion is here'''
    if xarm.ip==XARM6_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.12]) 
    elif xarm.ip==XARM6LEFT_IP:
        offset_grasp_in_object_frame = np.array([0, 0, +0.11]) # 物体局部坐标系下向下偏移
    if policy == "top_down":
        center_grasp = grasp_pose.copy()
        center_grasp[:3, 3] = grasp_pose[:3,3]- offset_grasp_in_object_frame
        print(f"center_grasp: {center_grasp}")
    elif policy=="z_axis":
        center_grasp = grasp_pose.copy()
        center_grasp[:3, :3] = grasp_pose[:3,:3]@ R.from_euler('z', 180, degrees=True).as_matrix()
        offset_grasp_in_world_frame = adjusted_rotation_matrix @ offset_grasp_in_object_frame
        center_grasp[:3, 3] = grasp_pose[:3,3] + offset_grasp_in_world_frame


    xarm.plan_and_execute(center_grasp)
    xarm.move_gripper(close =True, wait=True)

    '''grasping success, lift it up'''
    X_WorldEeflift =  center_grasp.copy()
    X_WorldEeflift[:3, 3] += adjusted_rotation_matrix @ np.array([0, 0, -0.15])
    xarm.plan_and_execute(X_WorldEeflift)
    xarm.move_to_home()
        
def specify_a_base_and_get_target_pose(object_name = None, arm_ip = XARM6LEFT_IP,  base_pose_cam_canonical= None, target_pose_cam_canonical = None):
    global mat_left_to_right
    reverse_xy = None
    reverse_xz = None
    symetric = None
    if arm_ip == XARM6LEFT_IP:
        sv = sv_left
    else:
        sv = sv_right
    '''here we get all the target pose based on the left arm base'''
    # sv = viser.ViserServer()
    base_pcd_world, pose, _ = get_object_pc_fp(object_name=object_name,arm_ip=arm_ip)
    # sv.scene.add_point_cloud("base_pcd", points = np.asarray(base_pcd_world.points), colors = (0,255,0),point_size = 0.002, point_shape = 'circle')
    if step == 2:
        reverse_xy = True
    if step == 6:
        reverse_xy = True
    base_pcd_world_canonical, base_mat, base_center = PointCloudUtils.canonicalize_point_cloud(base_pcd_world)
    base_pcd_world_canonical = np.asarray(base_pcd_world_canonical.points)
    # base_pcd_world_canonical, base_mat, base_center = PointCloudUtils.canonical_bbo(base_pcd_world, symetric= symetric, visualize= False,reverse_xy= reverse_xy,reverse_xz=reverse_xz)
    sv.scene.add_point_cloud("canonical_pcd", points=base_pcd_world_canonical, colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    sv.scene.add_frame("canonical_pose", wxyz=R.from_matrix(base_mat.T).as_quat()[[3, 0, 1, 2]], position=base_center, axes_length=0.3, axes_radius=0.01)
    base_pose_world_canonical = np.eye(4)
    base_pose_world_canonical[:3, :3] = base_mat.T @ R.from_euler('y' , 180, degrees= True).as_matrix()
    base_pose_world_canonical[:3, 3] = base_center
    canonical_transform_base_cam = base_pose_world_canonical @ np.linalg.inv(base_pose_cam_canonical)
    '''' get taget pose based on the canonical pose and assume the base in the real-world is static'''
    
    # for i in range(1,len(rotation_mats)):
    # target_pose_world_canonical = np.eye(4)
    # # in the box case, the rotation definition is little bit different from the stool
    # target_pose_world_canonical[:3, :3] = rotation_mats[0]
    # target_pose_world_canonical[:3, 3] = centers[0]
    for target_pose_cam_canonical in target_pose_cam_canonical:
        target_pose_world_canonical = canonical_transform_base_cam @ target_pose_cam_canonical

        global target_left, target_right
        target_left.append(target_pose_world_canonical)
        
        # print(f"mat_left_to_right: {mat_left_to_right}")
    for mat in target_left:
        target_right.append(mat_left_to_right @ mat)


    return base_pcd_world, target_left, target_right
    # print(f"target_right: {target_right}")


def move_to_ee_target_from_initial(object_name = 'box01', target_pose = None, xarm = None, base_pcd =  None, switch_to_right = False):
    
    symetric = True
    reverse_xy = None
    reverse_xz = True
    if switch_to_right:
        base_pcd = PointCloudUtils.transform_point_cloud(base_pcd, mat_left_to_right)
    status = False
    while not status:
        reverse_xy = not reverse_xy
        initial_pcd, initial_pose, _ = get_object_pc_fp(object_name = object_name, arm_ip = xarm.ip)
        xarm_right.update_attach(object_name=object_name, pose = initial_pose)
        xarm_right.update_collision_pcd(base_pcd, name='collision')
        cano_pcd, rotation_mat, center = PointCloudUtils.canonicalize_point_cloud(initial_pcd)
        if rotation_mat[2,2]>0:
            rotation_mat = rotation_mat @ R.from_euler('y' , 180, degrees= True).as_matrix()
        cano_pcd = np.asarray(cano_pcd.points)
        # cano_pcd, rotation_mat, center = PointCloudUtils.canonical_bbo(initial_pcd, reverse_xz = reverse_xz, visualize = False, symetric=symetric, reverse_xy=reverse_xy)
        pcd_handler1 = sv_right.scene.add_point_cloud("cano_pcd", points = cano_pcd, colors = (0,255,0),point_size = 0.001, point_shape = 'circle')
        # pcd_handler2 = sv_right.scene.add_point_cloud("ini_pcd", points = np.asarray(initial_pcd.points), colors = (0,255,0),point_size = 0.001, point_shape = 'circle')
        sv_right.scene.add_frame("canonical_pose", wxyz=R.from_matrix(rotation_mat[:3,:3].T).as_quat()[[3, 0, 1, 2]], position=center, axes_length=0.3, axes_radius=0.01)
        
        # bp()
        initial_cano_pose = np.eye(4)
        initial_cano_pose[:3,:3] = rotation_mat.T
        initial_cano_pose[:3,3] = center

        # sv.scene.add_frame("canonical_pose", wxyz=R.from_matrix(initial_cano_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=initial_cano_pose[:3,3], axes_length=0.3, axes_radius=0.01)
        # bp()
        transfer_mat = target_pose @ np.linalg.inv(initial_cano_pose)
        sv_right.scene.add_frame("target_pose", wxyz=R.from_matrix(target_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_pose[:3,3], axes_length=0.3, axes_radius=0.01)

        '''get the posision of the xarm frank, not ee'''
        _, xarm_frank = xarm.arm.get_position()

        sv_right.scene.add_frame("xarm_ee", wxyz=R.from_matrix(xarm_frank[:3,:3]).as_quat()[[3, 0, 1, 2]], position=xarm_frank[:3,3], axes_length=0.3, axes_radius=0.01)

        ee_target_pose = transfer_mat @ xarm_frank
        print("ee_target_pose", ee_target_pose)
        sv_right.scene.add_frame("frank_target_pose", wxyz=R.from_matrix(ee_target_pose[:3,:3]).as_quat()[[3, 0, 1, 2]], position=ee_target_pose[:3,3], axes_length=0.3, axes_radius=0.01)
        xarm.move_to_home()
        status = xarm.plan_and_execute(ee_target_pose)
    
    # # Patch
    # if step == 1:
    #     xarm.plan_and_execute(ee_target_pose, z_axis_offset = 0.02)
    #     xarm.move_z_axis(distance = 0.025)
    # if step == 2:
    #     xarm.plan_and_execute(ee_target_pose,z_axis_offset = 0.02)
    #     # xarm.rotation(angle = 5)
    #     xarm.move_z_axis(distance = 0.04)
    # if step == 3:
    #     xarm.plan_and_execute(ee_target_pose,z_axis_offset = 0, y_axis_offset = -0.03)
    #     xarm.move_y_axis(distance = -0.03)
    # if step == 4 :
    #     xarm.plan_and_execute(ee_target_pose,y_axis_offset = 0.02)
    #     # xarm.move_down(distance = -0.01)
    #     xarm.move_y_axis(distance = 0.02)
    # if step == 5:
    #     xarm.plan_and_execute(ee_target_pose,y_axis_offset = 0.03)
    #     xarm.move_y_axis(distance = 0.03)
    # if step == 6:
    #     xarm.plan_and_execute(ee_target_pose, y_axis_offset = -0.02)
    #     xarm.move_y_axis(distance = -0.02)



    # xarm.move_down(distance = 0.02)
    pcd_handler1.remove()
    # pcd_handler2.remove()
    xarm.remove_collision_pcd(name = "collision")
    xarm.remove_attachment(name = object_name)
    return ee_target_pose


def get_canonical_pose_cam_from_file(step):
    file_path = f"/home/shaol/data/zjx/rw/matrix_list/{step}.npy"
    dicts = np.load(file_path, allow_pickle=True).item()
    base_pose_cam_canonical = dicts['SE3_base']
    base_pose_cam_canonical[:3, :3] = base_pose_cam_canonical[:3, :3].T
    target_pose_cam_canonical = dicts['SE3_tgt']
    target_pose_cam_canonical[:3, :3] = target_pose_cam_canonical[:3, :3].T
    base_name = dicts['mesh_name'][0]
    target_name = dicts['mesh_name'][1]
    return base_pose_cam_canonical, target_pose_cam_canonical, base_name, target_name

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

                  
if __name__ == "__main__":
    '''initialize'''
    initial = True  # only query GPT for assembly order in the first time.
    mask_list = []  # for order_planing
    bbox_list = []  # for order_planing
    collision_pcd = None
    target_list = []
    cam2leftbase = None
    cam2rightbase = None
    cam_K = None
    step = None
    target_right = []
    target_left = []
    sv_left = viser.ViserServer()
    sv_right = viser.ViserServer()
    cam_serial = "241122074374"
    cam_K_path = Path(f"third_party/xarm6/data/camera/{cam_serial}/K.npy")
    cam_K = np.load(cam_K_path)
    arm_cam_X_BaseCamera_path_r = Path(f"third_party/xarm6/data/camera/{cam_serial}/0107_excalib_capture00/optimized_X_BaseCamera.npy")
    cam2rightbase = np.load(arm_cam_X_BaseCamera_path_r)
    arm_cam_X_BaseCamera_path_l = Path(f"third_party/xarm6/data/camera/{cam_serial}/1219_excalib_capture00/optimized_X_BaseCamera.npy")
    cam2leftbase = np.load(arm_cam_X_BaseCamera_path_l)
    rotation_path = '/home/shaol/data/zjx/rw/data/shelf119/rotation_matrix.npy'
    center_path = '/home/shaol/data/zjx/rw/data/shelf119/center.npy'
    rotation_mats = PointCloudUtils.read_matrices_from_npy(rotation_path)
    print(f'rotation mats{rotation_mats}')

    centers = PointCloudUtils.read_matrices_from_npy(center_path)
    print(f'centers{centers}')
    mat_left_to_right = cam2rightbase@ np.linalg.inv(cam2leftbase)
    grippermount_data_dir = Path("data/data_urdf/robot/xarm_gripper/hand_open.obj")
    gripper_trimesh = trimesh.load_mesh(grippermount_data_dir).apply_scale(1.1)
    base_joint_values = {
        1: np.array([ 0.01047198, -1.23045712, -0.48502786, 0.01047198, 1.08047372, 1.57622777]),
        # 1: np.array([ 5.9, -71.4, -24.1, 0.9, 55.6, -87.6])/180*np.pi,
        2: np.array([ 0.04712389, -0.27925268, -1.37357412,  0.06283185,  1.57603231,  1.57603231 ]),
        3: np.array([ 0.2, -12.4, -76.5, -0.7,  83.2,  32 ])/180*np.pi,
        5: np.array([ 8.4, -15.5, -73.1, 0.9,  82.6,  192.2 ])/180*np.pi
    }
    # special_adjustment = {
    #     2: 'base_rotate_xy'
    # }
    '''init the xarm'''
    xarm6_pk = XArm6WOEE()
    xarm_left = XArmController(ip=XARM6LEFT_IP)
    xarm_right = XArmController(ip=XARM6_IP)
    last_base_pose = None
    env_pcd = enviroment_constraint(XARM6PlannerCfg)
    sv_right.scene.add_point_cloud("env", points=env_pcd, colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    xarm_left.planner.mplib_add_point_cloud(env_pcd, name="env")
    xarm_right.planner.mplib_add_point_cloud(env_pcd, name="env")

    # xarm_left.move_gripper(open = True, wait = True)
    # xarm_right.move_gripper(open = True, wait = True)
    # xarm_right.move_to_home()
    # xarm_left.move_to_home()
    
    pick(xarm=xarm_left, object_name = 'shelf01' , policy='box')
    pick(xarm=xarm_right, object_name = 'shelf02' , policy='shelf')
   
    first_base_joints = np.array([ 1.9, -16.1, -15.3, 1.1, -27.9, 91.2])/180*np.pi
    # # # first_base_joints = np.array([ 0.3, -10.6, -53.9, 0.7, 16.8, 91.2])/180*np.pi
    # # # first_base_joints = np.array([2.1, 15.9, -106.6, -0.7, 79.3 , 91.2 ])/180*np.pi
    xarm_left.move_by_joints_value(first_base_joints)
   
    base_pose_cam_canonical = np.eye(4)
    target_pose_cam_canonical1 = np.eye(4)
    target_pose_cam_canonical2= np.eye(4)
    base_pose_cam_canonical[:3, :3] = rotation_mats[0].T
    base_pose_cam_canonical[:3, 3] = centers[0]
    target_pose_cam_canonical1[:3, :3] = rotation_mats[1].T
    target_pose_cam_canonical1[:3, 3] = centers[1]
    target_pose_cam_canonical2[:3, :3] = rotation_mats[2].T
    target_pose_cam_canonical2[:3, 3] = centers[2]
    target_pose_cam_canonical = [target_pose_cam_canonical1, target_pose_cam_canonical2]
    base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(object_name='shelf01', arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[0][:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[0][:3, 3], axes_length=0.3, axes_radius=0.01)
    sv_left.scene.add_frame("target_left2", wxyz=R.from_matrix(target_left[1][:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[1][:3, 3], axes_length=0.3, axes_radius=0.01)
    # print(f"target_left is {target_left}, target right is {target_right}")

    move_to_ee_target_from_initial(object_name = 'shelf02', target_pose = target_right[0], xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)
    # xarm_right.move_to_home()
    input("press enter to continue")
    target_right[1][:3, :3] = target_right[1][:3, :3] @ R.from_euler('y', 180, degrees=True).as_matrix()
    pick(xarm = xarm_right, object_name = 'shelf02', policy='shelf')
    right_hand_joints = np.array([ -7.5, -44.7, -30.2, 0, 75.1, 79.5])/180*np.pi
    xarm_right.move_by_joints_value(right_hand_joints)
    move_to_ee_target_from_initial(object_name = 'shelf02', target_pose = target_right[1], xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)
    # bp()



    # # '''start excuting picking two objects'''


    # '''get from SDK mannually'''
    # # first_base_joints = np.array([-0.02094395, -1.27065863, 0.0418879, 0.09208208, -0.10821041, 1.52170894])
    step = 1
    # first_base_joints = np.array([ 0.01047198, -1.23045712, -0.48502786, 0.01047198, 1.08047372, 1.57622777])
    # # first_base_joints = np.array([ 0.03665191, -0.93724181, -0.44331363,  0.12042772,  0.5969026 ,  1.57603231 ])
    # xarm_left.move_by_joints_value(first_base_joints)

    # '''get the traget pose for both base'''
    # base_pose_cam_canonical = np.eye(4)
    # target_pose_cam_canonical = np.eye(4)
    # base_pose_cam_canonical[:3, :3] = rotation_mats[1]
    # base_pose_cam_canonical[:3, 3] = centers[1]
    # target_pose_cam_canonical[:3, :3] = rotation_mats[0]
    # target_pose_cam_canonical[:3, 3] = centers[0]
    # base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    # print(f"target left is{target_left}, target right is {target_right}")
    # sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    # '''add a viser for right arm'''
    # # xarm_right.move_down(distance = 0.1)
    # move_to_ee_target_from_initial(object_name = 'box01', target_pose = target_right, xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)
    step = 2
    # second_base_joints = np.array([ 0.04712389, -0.27925268, -1.37357412,  0.06283185,  1.57603231,  1.57603231 ])
    # xarm_left.move_by_joints_value(second_base_joints)
    # pick(xarm = xarm_right, object_name='box02')
    # base_pose_cam_canonical = np.eye(4)
    # target_pose_cam_canonical = np.eye(4)
    # base_pose_cam_canonical[:3, :3] = rotation_mats[0].T
    # base_pose_cam_canonical[:3, 3] = centers[0]
    # target_pose_cam_canonical[:3, :3] = rotation_mats[1].T
    # target_pose_cam_canonical[:3, 3] = centers[1]
    # base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(object_name='box01_02', arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    # sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    # move_to_ee_target_from_initial(object_name = 'box02', target_pose = target_right, xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)
    
    step = 3
    # base reverse xy is False, move to reverse xy is True
    # pick(xarm = xarm_right, object_name='box03')
    # base_pose_cam_canonical = np.eye(4)
    # target_pose_cam_canonical = np.eye(4)
    # base_pose_cam_canonical[:3, :3] = rotation_mats[0].T
    # base_pose_cam_canonical[:3, 3] = centers[0]
    # target_pose_cam_canonical[:3, :3] = rotation_mats[1].T
    # target_pose_cam_canonical[:3, 3] = centers[1]
    # base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(object_name='box01_02_03', arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    # sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    # move_to_ee_target_from_initial(object_name = 'box03', target_pose = target_right, xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)
    # pose_sta = np.array([[1, 0, 0, 0.2],
    #                     [0, -1, 0, 0],
    #                     [0, 0, -1, 0.4],
    #                     [0, 0, 0, 1]])
    # xarm_right.plan_and_execute(pose_sta)
    # _, t1 = xarm_right.arm.get_position()
    # _, t2 = xarm_right.arm.get_position_se3()
    # print(t1, t2)
    # bp()

    # step = 4
    # forth_base_joints = np.array([ 0.2, -12.4, -76.5, -0.7,  83.2,  32 ])/180*np.pi
    # xarm_left.move_by_joints_value(forth_base_joints)
    # xarm_right.move_to_home()
    # # base reverse xy is TRue
    # # pick(xarm=xarm_right, object_name='box04')
    # base_pose_cam_canonical = np.eye(4)
    # target_pose_cam_canonical = np.eye(4)
    # base_pose_cam_canonical[:3, :3] = rotation_mats[0].T
    # base_pose_cam_canonical[:3, 3] = centers[0]
    # target_pose_cam_canonical[:3, :3] = rotation_mats[1].T
    # target_pose_cam_canonical[:3, 3] = centers[1]
    # base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(object_name='box01_02_03_04', arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    # sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    # move_to_ee_target_from_initial(object_name = 'box04', target_pose = target_right, xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)

    step = 5
    #symetric is False
    # fifth_base_joints = np.array([ 12.2, 0.5, -87.6, 1.7,  84.1,  188.4 ])/180*np.pi
    # xarm_left.move_by_joints_value(fifth_base_joints)
    # xarm_right.move_to_home()
    # # pick(xarm=xarm_right, object_name='box03')
    # base_pose_cam_canonical = np.eye(4)
    # target_pose_cam_canonical = np.eye(4)
    # base_pose_cam_canonical[:3, :3] = rotation_mats[0].T
    # base_pose_cam_canonical[:3, 3] = centers[0]
    # target_pose_cam_canonical[:3, :3] = rotation_mats[1].T
    # target_pose_cam_canonical[:3, 3] = centers[1]
    # base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(object_name='box01_02_03_04_05', arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    # sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    # move_to_ee_target_from_initial(object_name = 'box03', target_pose = target_right, xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)

    step = 6
    # reverse xy is True
    # dict = np.load('/home/shaol/data/zjx/rw/matrix_list/6.npy', allow_pickle=True).item()
    # print(dict)
    # bp()
    # fifth_base_joints = np.array([ 12.2, 0.5, -87.6, 1.7,  84.1,  188.4 ])/180*np.pi
    # xarm_left.move_by_joints_value(fifth_base_joints)
    # xarm_right.move_to_home()
    # pick(xarm=xarm_right, object_name='box04')
    # base_pose_cam_canonical = np.eye(4)
    # target_pose_cam_canonical = np.eye(4)
    # base_pose_cam_canonical[:3, :3] = rotation_mats[0].T
    # base_pose_cam_canonical[:3, 3] = centers[0]
    # target_pose_cam_canonical[:3, :3] = rotation_mats[1].T
    # target_pose_cam_canonical[:3, 3] = centers[1]

    # base_pcd, target_left, target_right = specify_a_base_and_get_target_pose(object_name='box01_02_03_04_05_06', arm_ip=XARM6LEFT_IP, base_pose_cam_canonical= base_pose_cam_canonical, target_pose_cam_canonical = target_pose_cam_canonical)
    # sv_left.scene.add_frame("target_left", wxyz=R.from_matrix(target_left[:3,:3]).as_quat()[[3, 0, 1, 2]], position=target_left[:3, 3], axes_length=0.3, axes_radius=0.01)
    # move_to_ee_target_from_initial(object_name = 'box04', target_pose = target_right, xarm = xarm_right, base_pcd = base_pcd, switch_to_right  = True)


    # for i in range(2):
    #     # pick(arm_ip=XARM6LEFT_IP)
    #     insert(arm_ip=XARM6LEFT_IP)
    #     # pick(arm_ip=XARM6_IP)
    #     # insert(arm_ip=XARM6_IP)


    

        