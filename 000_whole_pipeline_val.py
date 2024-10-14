import cv2
import pytorch3d.ops
import viser 

import numpy as np
import open3d as o3d
import hydra
import torch
import sys
import os
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
from xarm6_interface.utils.realsense import MultiRealsense, get_masked_pointcloud, remove_outliers
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from xarm6_interface import SAM_TYPE, SAM_PATH
from xarm6_interface.utils.sam_prompt_drawer import SAMPromptDrawer, shrink_mask
from xarm6_interface.utils.gpis import gpis_fit
from xarm6_interface.utils.mesh_and_urdf_utils import as_mesh
from xarm6_interface.utils.viser_utils import vis_vector
from diff_robot_hand.hand_model import LeapHandRight
from utils.hand_model import create_hand_model
from scipy.spatial.transform import Slerp
from model.network import create_network
from utils.multilateration import multilateration
from utils.hand_model import create_hand_model
from utils.optimization import *
from controller import controller
import pytorch3d
from matplotlib import pyplot as plt
from loguru import logger as lgr


def get_object_pc():
    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer", screen_scale=2.0, sam_checkpoint=SAM_PATH, device="cuda", model_type=SAM_TYPE)
    
    ''' multi realsense setup '''
    arm_left_cam_serial = "147122075879"
    arm_right_cam_serial = "241122074374"
    camera_serial_nums = [arm_left_cam_serial, arm_right_cam_serial]
    multi_rs = MultiRealsense(camera_serial_nums)
    arm_left_cam_K_path = Path("third_party/xarm6/data/camera/mounted_black/K.npy") 
    arm_left_cam_X_BaseCamera_path = Path("third_party/xarm6/data/camera/mounted_black/0914_excalib_capture1/optimized_X_BaseCamera.npy")
    arm_left_cam_K = np.load(arm_left_cam_K_path)
    arm_left_cam_X_BaseCamera = np.load(arm_left_cam_X_BaseCamera_path)
    arm_right_cam_K_path = Path("third_party/xarm6/data/camera/mounted_white/K.npy")
    arm_right_cam_X_BaseCamera_path = Path("third_party/xarm6/data/camera/mounted_white/0914_excalib_capture3/optimized_X_BaseCamera.npy")
    arm_right_cam_K = np.load(arm_right_cam_K_path)
    arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)
    multi_rs.set_intrinsics(0, arm_left_cam_K[0, 0], arm_left_cam_K[1, 1], arm_left_cam_K[0, 2], arm_left_cam_K[1, 2])
    multi_rs.set_intrinsics(1, arm_right_cam_K[0, 0], arm_right_cam_K[1, 1], arm_right_cam_K[0, 2], arm_right_cam_K[1, 2])
    camera_wxyzs = [
        R.from_matrix(arm_left_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
        R.from_matrix(arm_right_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    ]
    camera_positions = [arm_left_cam_X_BaseCamera[:3, 3],
                          arm_right_cam_X_BaseCamera[:3, 3]]
    X_BaseCamera_list = [arm_left_cam_X_BaseCamera, arm_right_cam_X_BaseCamera]
    
    ''' seg '''
    # stablize the camera
    for i in range(50):
        multi_rs.getCurrentData()
    
    rtr_dict_list = multi_rs.getCurrentData()
    input_boxes_list = []
    rgb_list = []
    mask_np_list = []
    shrink_mask_list = []
    masked_pc_o3d_list = []
    for camera_idx in range(len(rtr_dict_list)):
        rtr_dict = rtr_dict_list[camera_idx]
        rgb = rtr_dict["rgb"]
        depth = rtr_dict["depth"]
        pc_o3d = rtr_dict["pointcloud_o3d"]
        prompt_drawer.reset()
        mask_np = prompt_drawer.run(rgb)
        shrink_mask_np = shrink_mask(mask_np, shrink_coefficient=0.8)
        rgb_list.append(rgb)
            
        h, w = mask_np.shape[-2:]
        mask_np_list.append(mask_np)
        shrink_mask_list.append(shrink_mask_np)
        masked_pc_o3d = get_masked_pointcloud(rgb, depth, shrink_mask_np.reshape(h,w), multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"])
        X_BaseCamera = X_BaseCamera_list[camera_idx]
        rs_pc_in_C_np = np.asarray(masked_pc_o3d.points) 
        rs_pc_in_B = X_BaseCamera @ np.vstack([rs_pc_in_C_np.T, np.ones(rs_pc_in_C_np.shape[0])])
        rs_pc_in_B = rs_pc_in_B[:3].T
        masked_pc_o3d_W = o3d.geometry.PointCloud()
        masked_pc_o3d_W.points = o3d.utility.Vector3dVector(rs_pc_in_B)
        masked_pc_o3d_W.colors = masked_pc_o3d.colors
        masked_pc_o3d_list.append(masked_pc_o3d_W)
        
    # fusion the masked pc
    masked_pc_o3d_fusion = o3d.geometry.PointCloud()
    for masked_pc_o3d in masked_pc_o3d_list:
        masked_pc_o3d_fusion += masked_pc_o3d
    
    masked_pc_o3d_fusion = remove_outliers(masked_pc_o3d_fusion, z_min_thresh=0.02)    
    
    gpis, fitted_pcd = gpis_fit(masked_pc_o3d_fusion,)
    return fitted_pcd, masked_pc_o3d_fusion
    
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

@hydra.main(version_base="1.2", config_path="", config_name="validate")
def main(cfg):
    batch_size = cfg.dataset.batch_size
    device = torch.device(f'cuda:{cfg.gpu}')
    
    object_pc_o3d, masked_pc_o3d_fusion = get_object_pc()
    object_pc_np = np.asarray(object_pc_o3d.points)
    object_normals_np = np.asarray(object_pc_o3d.normals)
    object_pc_com = object_pc_np.mean(axis=0)  # (3,)
    object_pc = torch.tensor(object_pc_np-object_pc_com, dtype=torch.float32, device=device).unsqueeze(0)
    # downsample
    object_pc, object_selected_indices = pytorch3d.ops.sample_farthest_points(object_pc, K=512)
    object_pc = object_pc.repeat(batch_size, 1, 1)
    object_normals = torch.tensor(object_normals_np[object_selected_indices.flatten().detach().cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    
    network = create_network(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"epoch_28.pth", map_location=device))
    network.eval()
    robot_name = "leaphand"
    hand = create_hand_model(robot_name, device=device)
    
    initial_q = hand.get_canonical_q().to(device)
    initial_q = initial_q.repeat(batch_size, 1)
    initial_q[:, 3:6] = torch.from_numpy(get_inter_rot(batch_size)).to(device).to(torch.float32)
    robot_pc_list = []
    for i in range(batch_size):
        single_initial_q = initial_q[i]
        robot_pc = hand.get_transformed_links_pc(single_initial_q)[:, :3].unsqueeze(0)
        robot_pc_list.append(robot_pc)
    robot_pc = torch.cat(robot_pc_list, dim=0)
    
    with torch.no_grad():
        reldist = network(
            robot_pc,  # (B, N, 3)
            object_pc, # (B, M, 3)
        )['reldist'].detach()

        mlat_pc = multilateration(reldist, object_pc)  # (B, N, 3)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)  #
        optim_transform = process_transform(hand.pk_chain, transform)

        layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)  # (B, 22)
        outer_q, inner_q = controller(robot_name, predict_q)  
    
    
    import viser
    sv = viser.ViserServer()
    
    vis_hand = LeapHandRight(
        load_balls_urdf=True,
        load_visual_mesh=True,
        load_col_mesh=True,
        load_n_collision_point = 287,
    )
    L_dict = vis_hand.get_fc_grasp_loss(
            predict_q,
            object_pc,
            object_normals,
            L_fc_weight,
            L_distance_weight,
            L_object_pen_weight,
            L_self_pen_weight,
            L_joint_limit_weight,
            L_joint_ref_weight,
    )
    each_loss = L_dict["L_b"]
    lgr.info(f"each_loss: {each_loss}")
    
    # normalize each loss to [0, 1]
    each_loss_nmlzd = (each_loss - each_loss.min()) / (each_loss.max() - each_loss.min())
    each_loss_nmlzd = each_loss_nmlzd.detach().cpu().numpy()
    best_idx = np.argmin(each_loss_nmlzd)

    
    cmp = plt.get_cmap("rainbow")
    for i in range(batch_size):
    
        hand_mesh = as_mesh(vis_hand.get_hand_trimesh(predict_q[i], collision=True, visual=False)["collision"]).apply_translation(object_pc_com)
        hand_mesh_inner = as_mesh(vis_hand.get_hand_trimesh(inner_q[i], collision=True, visual=False)["collision"]).apply_translation(object_pc_com)
        hand_mesh_outer = as_mesh(vis_hand.get_hand_trimesh(outer_q[i], collision=True, visual=False)["collision"]).apply_translation(object_pc_com)
        hand_color = cmp(each_loss_nmlzd[i])[:3]
        hand_mesh.visual.vertex_colors = np.array([hand_color] * len(hand_mesh.visual.vertex_colors))
        hand_mesh_inner.visual.vertex_colors = np.array([hand_color] * len(hand_mesh_inner.visual.vertex_colors))
        hand_mesh_outer.visual.vertex_colors = np.array([hand_color] * len(hand_mesh_outer.visual.vertex_colors))

        if each_loss_nmlzd[i] == 0:
            hand_visual_mesh = vis_hand.get_hand_trimesh(predict_q[i], collision=False, visual=True)["visual"].apply_translation(object_pc_com)
            sv.scene.add_mesh_trimesh(f"hand_visual_{i}", hand_visual_mesh)
    
        sv.scene.add_mesh_trimesh(f"hand_{i}", hand_mesh, visible=False)
        sv.scene.add_mesh_trimesh(f"hand_inner_{i}", hand_mesh_inner, visible=False)
        sv.scene.add_mesh_trimesh(f"hand_outer_{i}", hand_mesh_outer, visible=False)
        
        
    sv.scene.add_point_cloud("masked_pc_o3d_fusion", np.asarray(masked_pc_o3d_fusion.points), colors=np.asarray(masked_pc_o3d_fusion.colors), point_size=0.001, point_shape="circle")
    sv.scene.add_point_cloud("gpis", np.asarray(object_pc_o3d.points), colors=np.asarray(object_pc_o3d.colors), point_size=0.001, point_shape="circle")
    
    for i in range(10):
        object_point = object_pc_np[i]
        object_point_normal = object_normals_np[i]
        sv.scene.add_mesh_trimesh(f"object_point_{i}", vis_vector(object_point, object_point_normal, ))    
        
    while True:
        time.sleep(1)
        
    
if __name__ == "__main__":
    main()
    

        