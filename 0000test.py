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
# def is_se3(pose, tol=1e-6):
#     # 检查是否是 4x4 矩阵
#     if not isinstance(pose, np.ndarray) or pose.shape != (4, 4):
#         return False
#     # 提取旋转矩阵部分和平移向量部分
#     R = pose[:3, :3]  # 旋转部分
#     t = pose[:3, 3]   # 平移部分
#     bottom_row = pose[3, :]  # 第 4 行
#     # 检查旋转部分是否正交: R^T R = I
#     if not np.allclose(np.dot(R.T, R), np.eye(3), atol=tol):
#         return False
#     # 检查旋转矩阵的行列式是否为 1
#     if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
#         return False
#     # 检查第 4 行是否为 [0, 0, 0, 1]
#     if not np.allclose(bottom_row, np.array([0, 0, 0, 1]), atol=tol):
#         return False
#     return True

# pts = np.random.rand(100, 3)

# canonical2 = np.array([[-0.0868393  ,-0.96479252  ,0.24826263  ,0.05078054],
#  [-0.99615597  ,0.08697041 ,-0.01046105  ,0.30935981],
#  [ 0.01149876  ,0.24821673  ,0.96863628 ,-0.04524611],
#  [ 0.          ,0.          ,0.          ,1.        ]])



# mat2 = np.array([[-0.0868393 , -0.99615597  ,0.01149876],
#  [-0.96479252 , 0.08697041 , 0.24821673],
#  [ 0.24826263 ,-0.01046105  ,0.96863628]])
# center2 = np.array([0.31310064 ,0.03331837 ,0.03445634])

# T1 = np.eye(4)
# T1[:3,:3] = mat2
# T1[:3,3] = center2
# print(mat2.T @ mat2)
# print(np.linalg.det(mat2))

# print(canonical2[:3, :3].T @ canonical2[:3, :3])
# print(np.linalg.det(canonical2[:3, :3]))
 
# T2 = np.linalg.inv(T1)
# print(np.allclose(T2[:3,:3], mat2.T))
# print(np.allclose(T2[:3,3], -mat2.T @ center2))
# print(np.isclose(T2, canonical2))

# print(T2[:3,:3].T @ T2[:3,:3])
# print(np.linalg.det(T2[:3,:3]))
# pts1 = (pts - center2) @ mat2.T
# pts1 = pts @ mat2.T - mat2.T @ center2
# pts2 = pts @ T2[:3, :3] + T2[:3, 3]

# print(np.allclose(pts1, pts2))
sv = viser.ViserServer()

def create_rectangular_point_cloud(length, width, height, num_points):
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
    x_coords = np.random.uniform((-length / 2)+0.5, (length / 2) + 0.5, num_points)
    y_coords = np.random.uniform((-width / 2)+0.5, (width / 2)+0.5, num_points)
    z_coords = np.random.uniform(-height / 2, height / 2, num_points)

    # Combine coordinates into a single array
    points = np.vstack((x_coords, y_coords, z_coords)).T

    # Create Open3D point cloud object
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)

    return point_cloud, points

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


# 使用PCA对批量点云进行规范化
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


# rec pts
length, width, height = 1, 0.5, 0.3  # Dimensions of the rectangular prism
num_points = 10000  # Number of points in the point cloud
rectangular_pcd, points= create_rectangular_point_cloud(length, width, height, num_points)
file_path = '/home/shaol/data/zjx/rw/data/16/point_cloud.npy'
test_pcd_path = '/home/shaol/data/zjx/rw/original_part_00_pcd.npy'
data_test = np.load(test_pcd_path, allow_pickle=True)  # 加载数据
data = np.load(file_path, allow_pickle=True)  # 加载数据
file_path = "/home/shaol/data/zjx/rw/data/16/rotation_matrix.npy"  # Replace with the actual path
matrices = read_matrices_from_npy(file_path)
rotation_board = matrices[0]
rotation_stick = matrices[1]
center_path = "/home/shaol/data/zjx/rw/data/16/center.npy"
center = read_matrices_from_npy(center_path)
# Print the loaded matrices
center_board = center[0]
center_stick= center[1]
# rw_pcd1, rw_pose1 = get_object_pc_fp(object_name='banzi',arm_ip=arm_ip)
# rw_pcd2,rw_pose2 = get_object_pc_fp(object_name='sticker',arm_ip=arm_ip)
'''big one, which can work in 1.4'''
# rotation_board = np.array([[-0.4921267497327684, -0.2913140084066984, -0.8203337191067326],
#    [-0.22280670350632997, -0.8688060453115218, 0.44219139351958964],
#    [-0.8415274416619484, 0.4003900649818579, 0.3626559813290277]])

# rotation_stick = np.array([[0.856986487985152, -0.4206289203636935, -0.29773389253584903],
#    [0.2218497718077104, 0.822592758608726, -0.5235683644313829],
#    [0.4651417398597355, 0.38263881773568165, 0.798267309240519]])

# center_board = np.array([-0.4408699379422127, 0.22284933888581338, -0.8093282272747179])
# center_stick = np.array([-0.1136382496693123, 0.10675322985117357, -0.6696112647913389])
# rotation_board = np.array([[-0.5405056484827092, -0.06810160384462882, -0.8385796417228811],
#    [0.043636164817601505, -0.9976462920720225, 0.05289386575923283],
#    [-0.8402080272636635, -0.008002966245953314, 0.5422051488624663]],)

# rotation_stick = np.array ([[0.8577353388106562, 0.1421954354157402, -0.4940349650604671],
#    [0.06223828964711993, 0.9251959348750065, 0.3743512753986076],
#    [0.5103101839638026, -0.3518422092894643, 0.7847232479703524]])

# center_board = np.array([-0.009378419689279759, 0.010161820934592483, -0.48465102854299])
# center_stick = np.array([-0.13619961128037467, -0.13515896045905232, -0.43009132697357266])

# rotation_board = np.array([[0.17096494693137942, -0.9806898787217053, 0.09496498667171072],
#    [-0.8395365563847709, -0.19544612040810916, -0.5069311437572367],
#    [0.515702780111133, 0.006940878202825962, -0.8567394392669345]])
# rotation_stick = np.array( [[-0.6300832441921064, 0.025479866586544886, 0.7761094521954247],
#    [0.7523088168222664, -0.2276358291240978, 0.6182340765684683],
#    [0.19242284043177524, 0.9734129163400886, 0.12426079342435847]])
# center_board = np.array([0.0013118662443665418, -0.006539438501826167, -0.4804468749087071])
# center_stick = np.array([-0.19457587982570668, -0.003250302227608983, -0.4380714407843407])
# 检查数据形状
print(f"Data shape: {data.shape}")
if data.shape[0] != 1 or data.ndim != 4 or data.shape[-1] != 3:
    raise ValueError(f"Expected a 4D array with shape (1, M, N, 3), but got shape {data.shape}.")

# 去掉第一维度，变为 (2, 1000, 3)
data = data[0]

# 遍历每个点云并计算质心
centroids = []
pcd = []
for i, points in enumerate(data):
    # 检查每个点云的形状
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Point cloud at index {i} is not valid. Expected shape (N, 3), got {points.shape}.")
    
    # 计算质心
    pcd.append(points)
    centroid = np.mean(points, axis=0)
    centroids.append(centroid)
    
    print(f"Point cloud {i + 1}:")
    print(f"  Shape: {points.shape}")
    print(f"  Centroid: {centroid}")

# 打印所有质心
print("\nAll centroids:")
for i, centroid in enumerate(centroids):
    print(f"Point cloud {i + 1} centroid: {centroid}")

pcd_1 = pcd[0]
pcd_2 = pcd[1]
print(f"Point cloud 1 shape: {pcd_1.shape}")
print(f"Point cloud 2 shape: {pcd_2.shape}")
# output_file = "groundtruth.npy"
# np.save(output_file, pcd_1)
# print(f"Point cloud saved to {output_file}")
# restore_origin = pcd_1 @ rotation_board + center_board  
# recanonical_pcd1, rotaion1,center1 = canonicalize_point_cloud(pcd_1)
# recanonical_pcd2, rotaion2,center2 = canonicalize_point_cloud(pcd_2)
# points_recanonical1 = np.asarray(recanonical_pcd1.points)
# points_recanonical2 = np.asarray(recanonical_pcd2.points)
pcd_restore = pcd_1 @ rotation_board + center_board
pcd_restore2 = pcd_2 @ rotation_stick + center_stick
restored_pcd = o3d.geometry.PointCloud()
restored_pcd.points = o3d.utility.Vector3dVector(pcd_restore)
restored_pcd2 = o3d.geometry.PointCloud()
restored_pcd2.points = o3d.utility.Vector3dVector(pcd_restore2)
sv.scene.add_point_cloud("board_pcd",points = np.asarray(restored_pcd.points),colors=(0,255,0),point_size=0.002,point_shape="circle")
sv.scene.add_point_cloud("stick_pcd",points = np.asarray(restored_pcd2.points),colors=(255,0,0),point_size=0.002,point_shape="circle")
sv.scene.add_frame("board_pose", wxyz=R.from_matrix(rotation_board.T).as_quat()[[3, 0, 1, 2]], position=center_board, axes_length=0.3, axes_radius=0.01)
sv.scene.add_frame("stick_pose", wxyz=R.from_matrix(rotation_stick.T).as_quat()[[3, 0, 1, 2]], position=center_stick, axes_length=0.3, axes_radius=0.01)
bp()
# load pts
# stick = np.loadtxt("rw_pcd2.txt")
# rectangular_pcd = o3d.geometry.PointCloud()
# rectangular_pcd.points = o3d.utility.Vector3dVector(stick)

for i in range(5):
    random_R = random_rotation_matrix()
    random_t = np.random.rand(3)
    new_point_cloud = o3d.geometry.PointCloud()
    rotated_pcd = np.asarray(data_test) @ random_R + random_t
    new_point_cloud.points = o3d.utility.Vector3dVector(rotated_pcd)
    canonicalized_pcd, rotation_matrix0, centroid0 = canonicalize_point_cloud(new_point_cloud)
    print(np.linalg.det((rotation_matrix0)))
    sv.scene.add_point_cloud(f"pts{i}",points = np.asarray(new_point_cloud.points),colors=(0,255,0),point_size=0.002,point_shape="circle")
    # sv.scene.add_point_cloud("after",points = np.asarray(canonicalized_pcd2.points),colors=(255,0,0),point_size=0.002,point_shape="circle")
    # sv.scene.add_frame("canonical2", wxyz=R.from_matrix(canonical2[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical2[:3, 3], axes_length=0.03, axes_radius=0.001)
    sv.scene.add_frame(f"PCA{i}", wxyz=R.from_matrix(rotation_matrix0[:3, :3].T).as_quat()[[3, 0, 1, 2]], position=centroid0, axes_length=0.3, axes_radius=0.01)
input("Press Enter to continue...")

# canonicalized_pcd, rotation_matrix0, centroid0 = canonicalize_point_cloud(rectangular_pcd)
# canonical2 = np.eye(4)
# canonical2[:3, :3] = rotation_matrix0.T
# canonical2[:3, 3] = -centroid0@rotation_matrix0.T

# restore = np.asarray(canonicalized_pcd.points) @ canonical2[:3, :3] + canonical2[:3, 3]
# print("centroid0:", centroid0)
# print("Bounding box of the point cloud:", np.min(points, axis=0), np.max(points, axis=0))

# canonicalized_pcd2 = o3d.geometry.PointCloud()
# canonicalized_pcd2.points = o3d.utility.Vector3dVector(restore)
# sv.scene.add_point_cloud("origin",points = np.asarray(canonicalized_pcd.points),colors=(0,255,0),point_size=0.002,point_shape="circle")
# sv.scene.add_point_cloud("after",points = np.asarray(canonicalized_pcd2.points),colors=(255,0,0),point_size=0.002,point_shape="circle")
# sv.scene.add_frame("canonical2", wxyz=R.from_matrix(canonical2[:3, :3]).as_quat()[[3, 0, 1, 2]], position=canonical2[:3, 3], axes_length=0.03, axes_radius=0.001)
# sv.scene.add_frame("PCA", wxyz=R.from_matrix(rotation_matrix0[:3, :3]).as_quat()[[3, 0, 1, 2]], position=centroid0, axes_length=0.03, axes_radius=0.001)
# input("Press Enter to continue...")
# # Visualize the point cloud
# o3d.visualization.draw_geometries([rectangular_pcd])
# P1 = np.random.rand(100, 3)
# random_R = random_rotation_matrix()
# P2 = P1 @ random_R
# canonicalized_pcd, rotation_matrix0, centroid0 = canonicalize_point_cloud(P1)
# canonicalized_pcd, rotation_matrix1, centroid1 = canonicalize_point_cloud(P2)
# print(rotation_matrix0@random_R)
# print(rotation_matrix1)



# print(rotation_matrix0.T)
# print(random_R @ rotation_matrix1.T)