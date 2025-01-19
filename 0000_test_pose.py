import numpy as np
from utils import PointCloudUtils
import viser 
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from pdb import set_trace as bp
import trimesh  

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


sv = viser.ViserServer()
mesh = trimesh.load_mesh("/home/shaol/data/zjx/rw/object_mesh_new/shelf02/shelf02.obj")
# initial_p = np.load("pts1.npy")
sampled_points = mesh.sample(10000)  
initial_pcd = o3d.geometry.PointCloud()
initial_pcd.points = o3d.utility.Vector3dVector(sampled_points)
for i in range(100):
    sampled_points = mesh.sample(10000)   
    random_R = random_rotation_matrix()
    random_t = np.random.rand(3)
    new_point_cloud = o3d.geometry.PointCloud()
    rotated_pcd = np.asarray(sampled_points) @ random_R + random_t
    new_point_cloud.points = o3d.utility.Vector3dVector(rotated_pcd)
    cano_pcd, rotation_mat, center = PointCloudUtils.canonical_bbo(new_point_cloud, visualize = False)
    print(np.linalg.det(rotation_mat))
    # print(f"cano_pcd is {cano_pcd}, rotation mat is {rotation_mat}, center is {center}")
    # print("Check Orthogonality:\n", np.dot(rotation_mat, rotation_mat.T))
    restore_pcd = cano_pcd @ rotation_mat + center
    # rereconanical = (restore_pcd-center) @ rotation_mat.T
    # rereconanical = (np.asarray(new_point_cloud.points) - center) @ rotation_mat.T
    # sv.scene.add_point_cloud(f"rereconanical{i}", points = rereconanical, colors = (0,255,0),point_size = 0.01, point_shape = 'circle')

    sv.scene.add_point_cloud(f"restore_pcd{i}", points = restore_pcd, colors = (0,0,255),point_size = 0.001, point_shape = 'circle')
    target_pose = np.eye(4)
    target_pose[:3, :3] = rotation_mat.T
    target_pose[:3, 3] = center
    sv.scene.add_point_cloud(f"cano_pcd{i}", points = cano_pcd, colors = (0,255,0),point_size = 0.001, point_shape = 'circle')
    sv.scene.add_point_cloud(f"ini_pcd{i}", points = np.asarray(new_point_cloud.points), colors = (0,255,0),point_size = 0.001, point_shape = 'circle')
    # sv.scene.add_frame("canonical_pose2", wxyz=R.from_matrix(target_pose[:3, :3].T).as_quat()[[3, 0, 1, 2]], position=target_pose[:3, 3], axes_length=0.3, axes_radius=0.01)
    sv.scene.add_frame(f"canonical_pose{i}", wxyz=R.from_matrix(target_pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=target_pose[:3, 3], axes_length=0.3, axes_radius=0.01)
bp()