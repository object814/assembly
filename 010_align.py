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
X_BaserightBaseleft = np.array([
        [-0.9994789, -0.02021456, -0.02516249, 1.1056788],
        [0.02050998, -0.99972314, -0.01153819, -0.03005717],
        [-0.02492228, -0.01204827, 0.9996168, 0.00512242],
        [0.0, 0.0, 0.0, 1.0]
    ])
'''https://drive.google.com/drive/folders/1g6DQnhj8BTBELwc1hyeRU1EXFZmvxcHB?usp=sharing'''
def read_matrices_from_npy(file_path):
    """
    Read matrices from a .npy file.

    Parameters:
        file_path (str): Path to the .npy file.

    Returns:
        list: A list of matrices loaded from the file.
    """
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

# Example usage
file_path = "/home/shaol/data/zjx/rw/data/15_1/rotation_matrix.npy"  # Replace with the actual path
matrices = read_matrices_from_npy(file_path)
rotation_board = matrices[0]
rotation_stick = matrices[1]
center_path = "/home/shaol/data/zjx/rw/data/15_1/center.npy"
center = read_matrices_from_npy(center_path)
# Print the loaded matrices
center_board = center[0]
center_stick= center[1]
print(f"rotation_board: {rotation_board}")
print(f"rotation_stick: {rotation_stick}")
print(f"center_board: {center_board}")
print(f"center_stick: {center_stick}")
bp()

'''https://drive.google.com/drive/folders/1fXAIprK1tRvN_JbT5SLc-hnquFccQkoQ?usp=sharing'''
# rotation_board = np.array([[-0.5405056484827092, -0.06810160384462882, -0.8385796417228811],
#    [0.043636164817601505, -0.9976462920720225, 0.05289386575923283],
#    [-0.8402080272636635, -0.008002966245953314, 0.5422051488624663]],)

# rotation_stick = np.array ([[0.8577353388106562, 0.1421954354157402, -0.4940349650604671],
#    [0.06223828964711993, 0.9251959348750065, 0.3743512753986076],
#    [0.5103101839638026, -0.3518422092894643, 0.7847232479703524]])

# center_board = np.array([-0.009378419689279759, 0.010161820934592483, -0.48465102854299])
# center_stick = np.array([-0.13619961128037467, -0.13515896045905232, -0.43009132697357266])

# mat_for_stick = np.array([[-0.8072, -0.1237, -0.5772, -0.3197],
#                           [0.1362, 0.9124, -0.3860, -0.2194],
#                           [0.5744, -0.3902, -0.7196, -1.1432],
#                           [0.0, 0.0, 0.0, 1.0]])

# mat_for_board = np.array([[0.7562, -0.2846, -0.5892, -0.0353],
#                           [0.4688, 0.8638, 0.1846,  -0.0648],
#                           [0.4564, -0.4158, 0.7866, -1.2246],
#                           [0.0, 0.0, 0.0, 1.0]])


# rotation_board = np.array([[-0.4921267497327684, -0.2913140084066984, -0.8203337191067326],
#    [-0.22280670350632997, -0.8688060453115218, 0.44219139351958964],
#    [-0.8415274416619484, 0.4003900649818579, 0.3626559813290277]])

# rotation_stick = np.array([[0.856986487985152, -0.4206289203636935, -0.29773389253584903],
#    [0.2218497718077104, 0.822592758608726, -0.5235683644313829],
#    [0.4651417398597355, 0.38263881773568165, 0.798267309240519]])

# center_board = np.array([-0.4408699379422127, 0.22284933888581338, -0.8093282272747179])
# center_stick = np.array([-0.1136382496693123, 0.10675322985117357, -0.6696112647913389])
trans_cam_world = np.zeros((4, 4))
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


def create_rectangular_point_cloud(length, width, height, num_points, center=(0, 0, 0.5)):
    """
    Create a point cloud in the shape of a rectangular prism.
    
    Args:
        length (float): Length of the rectangular prism along the x-axis.
        width (float): Width of the rectangular prism along the y-axis.
        height (float): Height of the rectangular prism along the z-axis.
        num_points (int): Number of points in the point cloud.

    Returns:
        o3d.geometry.PointCloud: The rectangular point cloud.
    """
    # Randomly generate points within the rectangular prism
    x_coords = np.random.uniform((-length / 2)+0.5, (length / 2) + 0.5, num_points) + center[0]
    y_coords = np.random.uniform((-width / 2)+0.5, (width / 2)+0.5, num_points)+ center[1]
    z_coords = np.random.uniform(-height / 2, height / 2, num_points)+ center[2]

    # Combine coordinates into a single array
    points = np.vstack((x_coords, y_coords, z_coords)).T

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud, points

def get_pcd(object_name):
    mesh = trimesh.load(f"object_mesh_new/{object_name}/{object_name}.obj")

def sample_points_from_mesh(mesh, num_points, seed=None):
    if seed is not None:
        np.random.seed(seed)  # 设置随机数种子
    points, _ = trimesh.sample.sample_surface(mesh, num_points)
    return points
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
    if np.linalg.det(rotation_matrix) < 0:
        rotation_matrix = -rotation_matrix
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
            # normals = mesh.face_normals[face_indices]
            '''points are sampled here'''
            points = sample_points_from_mesh(mesh, 50000, seed=0)
            object_pc_o3d = o3d.geometry.PointCloud()
            object_pc_o3d.points = o3d.utility.Vector3dVector(points)
            # object_pc_o3d.normals = o3d.utility.Vector3dVector(normals)
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
            
            points, face_indices = mesh.sample(1000, return_index=True)
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
# object_name = 'blue_cup' # 3/5 ; 7/10
# object_name = 'ginger_cookies_box' # 5/5 ; 10/10
# object_name = 'camera_bag' # 5/5 ; 10/10 
# object_name = 'tea_box'#  4/5 ; 8/10
# object_name = 'brush'  # 4/5 ; 9/10
# object_name = 'milk_box'
# object_name = 'ps_controller'
# object_name = 'triangle_cad' 
# object_name = 'qianzi'
# object_name = 'zhijia'
# object_name = 'apple' # 4/5 ; 9/10 # ok
# object_name = 'gun' # ok
# object_name = 'stage' # ok
# object_name = 'hand_grips' # woliqi
# object_name = 'light_blue_cup' # no
# object_name = 'angle_iron' # no fail once
# object_name = 'whiteboard_pen'
# object_name = 'u_iron' # fail once
# object_name = 'clipper'
# object_name = 'blue_cup'
# object_name = 'gripper'
# object_name = 'spatula'
# object_name = 'chu'
# object_name = "wheel"



planner_timestep = 1.0 / 50.0
cmd_timestep = 1.0 / 100.0 
pregrasp_retreat_distance = 0.08
# @hydra.main(version_base="1.2", config_path="", config_name="validate")

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
        rotation_angle = np.pi 
        rotation_matrix = np.eye(4)
        rotation_matrix[:3, :3] = np.array([
            [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
            [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
            [0,                     0,                      1]
        ])
        pose = pose @ rotation_matrix

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
    # canonicalized_pcd, rotation_matrice, centroid = canonicalize_point_cloud(object_pc_o3d)
    mat_A = X_WorldObject
    

    # sv.scene.add_point_cloud("canonical_pc", points=np.asarray(canonicalized_pcd.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")
    # sv.scene.add_frame("canonical_pose", wxyz=R.from_matrix(rotation_matrice[:3, :3]).as_quat()[[3, 0, 1, 2]], position=centroid, axes_length=0.03, axes_radius=0.001)
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

    def is_se3(pose, tol=1e-6):
        # 检查是否是 4x4 矩阵
        if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
            return False

        # 提取旋转矩阵部分和平移向量部分
        R = pose[:3, :3]  # 旋转部分
        t = pose[:3, 3]   # 平移部分
        bottom_row = pose[3, :]  # 第 4 行

        # 检查旋转部分是否正交: R^T R = I
        if not np.allclose(np.dot(R.T, R), np.eye(3), atol=tol):
            return Falserotation_stick

        # 检查第 4 行是否为 [0, 0, 0, 1]
        if not np.allclose(bottom_row, np.array([0, 0, 0, 1]), atol=tol):
            return False

        return True

    def get_pre_pose_from_target_pose(target_pose, ee_offset=0.166):
        ee_offset = ee_offset  # 末端执行器距离法兰的距离
        ee_offset_vector = np.array([0, 0, -ee_offset])  # 在目标坐标系下的偏移向量

        ee_target_pose = target_pose.copy()
        # 应用偏移
        ee_target_pose[:3, 3] += target_pose[:3, :3] @ ee_offset_vector.reshape(3)
        # ee_target_pose[:3, 3] -= ee_offset_grasp
        # 沿目标姿态的 x 轴偏移 10 cm
        x_axis_offset = 0.1  # 10 cm
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
    for i in range(5):
        rw_pcd1, rw_pose1 = get_object_pc_fp(object_name='original_part_00',arm_ip=arm_ip)
            # 假设 rw_pcd1 是一个 Open3D 点云对象
        points = np.asarray(rw_pcd1.points)  # 提取点云坐标

        # # 保存为 .npy 文件
        # output_file = "original_part_00_pcd.npy"
        # np.save(output_file, points)
        # print(f"Point cloud saved to {output_file}")

        
        # ''' read the gt'''
        # file_path = '/home/shaol/data/zjx/rw/groundtruth.npy'
        # gt  = np.load(file_path, allow_pickle=True)
        # gt_pcd  = o3d.geometry.PointCloud()
        # gt_pcd.points = o3d.utility.Vector3dVector(gt)
        # initial = False
        # rw_pcd2,rw_pose2 = get_object_pc_fp(object_name='sticker',arm_ip=arm_ip)
        canonical_pcd1, mat1, center1 = canonicalize_point_cloud_heu(points)
        # print(np.asarray(rw_pcd1))
        # canonical_pcd2, mat2, center2 = canonicalize_point_cloud(rw_pcd2)
        # np.savetxt("rw_pcd2.txt",np.asarray(rw_pcd2.points))
        # rectangular_pcd, points= create_rectangular_point_cloud(1,0.5,0.3,10000)
        # rw_pcd2 = rectangular_pcd
        # for i in range(5):
        #     random_R = random_rotation_matrix()
        #     new_point_cloud = o3d.geometry.PointCloud()
        #     rotated_pcd = np.asarray(rw_pcd2.points) @ random_R
        #     new_point_cloud.points = o3d.utility.Vector3dVector(rotated_pcd)
        #     canonicalized_pcd, rotation_matrix0, centroid0 = canonicalize_point_cloud(new_point_cloud)
        #     restored_pcd = np.asarray(new_point_cloud.points) @ rotation_matrix0
            
        #     # sv.scene.add_point_cloud("after",points = np.asarray(canonicalized_pcd2.points),colors=(255,0,0),point_size=0.002,point_shape="circle")

        #     sv.scene.add_point_cloud(f"pts{i}",points = np.asarray(new_point_cloud.points),colors=(0,255,0),point_size=0.002,point_shape="circle")
            
        #     # sv.scene.add_point_cloud("after",points = np.asarray(canonicalized_pcd2.points),colors=(255,0,0),point_size=0.002,point_shape="circle")
        #     # sv.scene.add_frame("canonical2", wxyz=R.from_matrix(canonical2[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical2[:3, 3], axes_length=0.03, axes_radius=0.001)
        #     sv.scene.add_frame(f"PCA{i}", wxyz=R.from_matrix(rotation_matrix0[:3, :3].T).as_quat()[[3, 0, 1, 2]], position=centroid0, axes_length=0.3, axes_radius=0.01)
        # bp()
        # if np.linalg.det(mat2) < 0:
        #     mat2[:, 0] *= -1
        canonical1 = np.eye(4)
        canonical1[:3, :3] = mat1.T
        # canonical1[:3, 3] = -center1@mat1.T
        # canonical1[:3, :3] = mat1
        canonical1[:3, 3] = center1
        
        # canonical2 = np.eye(4)
        # canonical2[:3, :3] = mat2.T
        # # canonical2[:3, 3] = -center2@mat2.T
        # # canonical2[:3, :3] = mat2
        # canonical2[:3, 3] = center2
        # # canonical1 = np.linalg.inv(canonical1)
        # # canonical2 = np.linalg.inv(canonical2)
        # print(f"mat2: {mat2}")
        # print(f"center2: {center2}")
        # print(f"canonical1: {canonical1}")
        # print(f"canonical2: {canonical2}")

        b1 = np.eye(4)
        s1 = np.eye(4)
        b1[:3,:3] = rotation_board.T
        b1[:3,3] = center_board
        s1[:3,:3] = rotation_stick.T
        s1[:3,3] = center_stick
        
        # mat_for_stickT = mat_for_stick[:3,:3].copy()
        # mat_for_boardT = mat_for_board[:3,:3].copy()

        # mat_for_stickT = mat_for_stickT.T
        # mat_for_boardT= mat_for_boardT.T
        # mat_for_stick[:3,:3] = mat_for_stickT
        # mat_for_board[:3,:3] = mat_for_boardT
        # print(f"mat_for_stick: {mat_for_stick}")
        # print(f"mat_for_board: {mat_for_board}")
        # target_pose =  canonical2 @ np.linalg.inv(mat_for_stick)@np.linalg.inv(canonical1@ mat_for_board)

        target_pose = canonical1 @np.linalg.inv(b1) @ s1
        # target_pose = canonical2 @ rw_pose2 
        # canon_pts2 = (np.asarray(rw_pcd2.points) - center2) @ mat2.T

        # canon_pts2 = np.asarray(canonical_pcd2.points) @ canonical2[:3, :3].T+ canonical2[:3, 3]
        # target_rot = mat2
        # target_center = center2
        # target2 = np.eye(4)
        # target2[:3, :3] = target_rot
        # target2[:3, 3] = target_center 
        # sv.scene.add_frame("object_pose", wxyz=R.from_matrix(canonical2[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical2[:3, 3], axes_length=0.3, axes_radius=0.01)

        # canon_pts2 = np.asarray(rw_pcd2.points)

        # canon_pts2 = np.concatenate([canon_pts2, np.ones((canon_pts2.shape[0], 1))], axis=1) @ canonical2

        # canon_pts2 = canon_pts2[:, :3] / canon_pts2[:, 3].reshape(-1, 1)
        # if is_se3(target_pose) :
        #     lgr.info("The target pose and the current pose are valid SE(3) matrices.")
        canonicalized_pcd = o3d.geometry.PointCloud()
        # canonicalized_pcd.points = o3d.utility.Vector3dVector(canon_pts2)
        # sv.scene.add_point_cloud("canon_pc",points = np.asarray(canonical_pcd2.points),colors=(0,255,0),point_size=0.002,point_shape="circle")
        sv.scene.add_point_cloud("gt_pc2",points = np.asarray(rw_pcd1.points),colors=(0,255,255),point_size=0.002,point_shape="circle")
        # sv.scene.add_point_cloud("pcd_board",points = np.asarray(rw_pcd1.points),colors=(0,255,0),point_size=0.002,point_shape="circle")
        sv.scene.add_frame("board_pose", wxyz=R.from_matrix(canonical1[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical1[:3, 3], axes_length=0.3, axes_radius=0.01)
        # sv.scene.add_frame("object_pose", wxyz=R.from_matrix(canonical2[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical2[:3, 3], axes_length=0.03, axes_radius=0.001)
        print(f"target_pose: {target_pose}")
        # sv.scene.add_point_cloud("object_pc", points=np.asarray(rw_pcd2.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
        # sv.scene.add_point_cloud("obstacle_pc", points=np.asarray(rw_pcd1.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
        sv.scene.add_frame("target_pose", wxyz=R.from_matrix(target_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=target_pose[:3, 3], axes_length=0.3, axes_radius=0.01)
        input("Press Enter to check the target pose...")
    '''' get a taget pose until is correct'''
    # correct_target_pose = False
    # while not correct_target_pose:
    #     # target_pcd,target_pose = get_object_pc_fp(object_name='sticker',arm_ip=arm_ip)
    #     x_axis = target_pose[:3, 0]
    #     z_x_axis = x_axis[2]
    #     if z_x_axis > 0:
    #         lgr.info("The target pose is incorrect, please adjust the pose.")
    #         continue
    #     else:
    #         lgr.info("The target pose is correct, please continue.")
    #         break
    mesh = trimesh.load_mesh("object_mesh_new/original_part_01/original_part_01.obj")
    target_pcd = sample_points_from_mesh(mesh, 1000)
    # target_center = get_pcd_center(target_pcd)
    object_pc_o3d = o3d.geometry.PointCloud()
    object_pc_o3d.points = o3d.utility.Vector3dVector(target_pcd)
    object_pc_o3d.transform(target_pose)
    print(f"target pose: {target_pose}")
    # sv.scene.add_point_cloud("target", points=np.asarray(object_pc_o3d.points), colors=(0, 255, 0), point_size=0.002, point_shape="circle")

    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    # xarm6_planner.mplib_add_point_cloud(np.asarray(collision_pcd.points), name="collision_pc")

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
    
    for i in range(1):
        # pick(arm_ip=XARM6LEFT_IP)
        insert(arm_ip=XARM6LEFT_IP)
        # pick(arm_ip=XARM6_IP)
        # insert(arm_ip=XARM6_IP)


    

        