import cv2
import pytorch3d.ops
import time
import viser 

import numpy as np
import open3d as o3d
import hydra
import torch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
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
from xarm6_interface.arm_rw import XArm6RealWorld
from diff_robot_hand.hand_model import LeapHandRight
from utils.hand_model import create_hand_model
from scipy.spatial.transform import Slerp
from model.network import create_network
from utils.multilateration import multilateration
from utils.hand_model import create_hand_model
from utils.optimization import *
from controller import controller
from xarm6_interface.arm_mplib import XARM6PlannerCfg, XARM6Planner, min_jerk_interpolator_with_alpha
import pytorch3d
from matplotlib import pyplot as plt
from loguru import logger as lgr
from leaphand_rw.leaphand_rw import LeapNode, leap_from_sim_to_rw
from xarm6_interface.envs.table_and_workspace_pc import WoodenTableMount, create_bounding_box_pc, create_plane_pc, env_pc_post_process
from xarm6_interface.arm_pk import XArm6WOEE
from xarm6_interface.utils.viser_utils import update_viser_mp_result
from FoundationPose.estimater import *

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

def count_break_down_motor(leaphand: LeapNode):
    
    motor_current_list = []
    for _ in range(10):
        motor_current_list.append(leaphand.read_cur())
        time.sleep(0.1)
    motor_current_list = np.array(motor_current_list)  # shape = (10, 16)
    
    motor_current_list = motor_current_list.T
    motor_current_list = np.abs(motor_current_list)
    motor_current_list = np.mean(motor_current_list, axis=1)
    num_break_down_motor = np.sum(motor_current_list < 0.01)
    break_down_motor_list = np.where(motor_current_list < 0.01)
    
    return num_break_down_motor, break_down_motor_list


def get_object_pc_fp():
    object_name = 'binoculars'
    object_name = 'dinosaur'  #5/5 ； 9/10
    # object_name = 'camera_bag' # 5/5 ; 10/10 
    # object_name = 'tea_box'#  4/5 ; 8/10
    # object_name = 'duck'  # 3/5 ; 8/10 
    # object_name = 'brush'  # 4/5 ; 9/10
    # object_name = 'blue_cup' # 3/5 ; 7/10
    # object_name = 'apple' # 4/5 ; 9/10
    object_name = 'toilet_cleaner'  # 5/5 ; 10/10
    # object_name = 'ginger_cookies_box' # 5/5 ; 10/10
    # object_name = 'rubic_cube'  #  4/5 ; 9/10
    
    # object_name = 'milk_box'
    # object_name = 'flashlight'
    # object_name = 'brown_bottle' 
    # object_name = 'iphone_box' # 2/5 ; 3/10
    # object_name = 'realsense_box' # 3/5 ; 5/10
    # object_name = 'ps_controller'
    # object_name = 'triangle_cad' 
    # object_name = 'fish'
    # object_name = 'qianzi'
    # object_name = 'gun'
    # object_name = 'stage'
    # object_name = 'zhijia'
    
    mesh = trimesh.load(f"object_mesh/{object_name}/{object_name}.obj")
    
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
    
    arm_right_cam_serial = "241122074374"
    camera_serial_nums = [arm_right_cam_serial]
    multi_rs = MultiRealsense(camera_serial_nums)
    arm_right_cam_K_path = Path("third_party/xarm6/data/camera/mounted_white/K.npy")
    arm_right_cam_K = np.load(arm_right_cam_K_path)
    arm_right_cam_X_BaseCamera_path = Path("third_party/xarm6/data/camera/mounted_white/1010_excalib_capture00/optimized_X_BaseCamera.npy")
    arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)
    multi_rs.set_intrinsics(0, arm_right_cam_K[0, 0], arm_right_cam_K[1, 1], arm_right_cam_K[0, 2], arm_right_cam_K[1, 2])
    camera_wxyzs = [
        R.from_matrix(arm_right_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    ]
    camera_positions = [arm_right_cam_X_BaseCamera[:3, 3]]
    X_BaseCamera_list = [arm_right_cam_X_BaseCamera]
    
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
        
        pose = est.register(K=arm_right_cam_K, rgb=rgb, depth=depth, ob_mask=mask_np, iteration=5)
        pose = arm_right_cam_X_BaseCamera @ pose
        
        points, face_indices = mesh.sample(512, return_index=True)
        normals = mesh.face_normals[face_indices]
        
        object_pc_o3d = o3d.geometry.PointCloud()
        object_pc_o3d.points = o3d.utility.Vector3dVector(points)
        object_pc_o3d.normals = o3d.utility.Vector3dVector(normals)
        object_pc_o3d.transform(pose)
        
        
        # server = viser.ViserServer(host='127.0.0.1', port=8080)
        
        # server.scene.add_frame(
        #     "object_frame",
        #     wxyz=trimesh.transformations.quaternion_from_matrix(pose),
        #     position=pose[:3, 3],
        #     axes_length=0.2,
        #     axes_radius=0.005,
        #     visible=True
        # )
        
        # server.scene.add_mesh_trimesh(
        #     'object_mesh',
        #     mesh,
        #     wxyz=trimesh.transformations.quaternion_from_matrix(pose),
        #     position=pose[:3, 3],
        # )
        
        # server.scene.add_point_cloud(
        #     'object_pc',
        #     np.asarray(object_pc_o3d.points),
        #     colors=(102, 192, 255),
        #     point_size=0.001,
        #     point_shape="circle"
        # )
        
        # server.scene.add_camera_frustum(
        #     f"rs_camera_img_{camera_idx}",
        #     fov=rtr_dict["fov_x"],
        #     aspect=rtr_dict["aspect_ratio"],
        #     wxyz=camera_wxyzs[0],
        #     position=camera_positions[0],
        #     image=rgb,
        #     scale=0.2
        # )
        
        # pc = np.asarray(pc_o3d.points)
        # pc = (arm_right_cam_X_BaseCamera[:3, :3] @ pc.T).T + arm_right_cam_X_BaseCamera[:3, 3]
        # server.scene.add_point_cloud(
        #     'scene',
        #     pc,
        #     colors=np.asarray(pc_o3d.colors),
        #     point_size=0.001,
        #     point_shape="circle"
        # )
        
        # while True:
        #     time.sleep(1)
        
        return object_pc_o3d

    
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

@hydra.main(version_base="1.2", config_path="", config_name="validate")
def main(cfg):
    batch_size = cfg.dataset.batch_size
    device = torch.device(f'cuda:{cfg.gpu}')
    
    # object_pc_o3d, masked_pc_o3d_fusion = get_object_pc()
    t1 = time.time()
    object_pc_o3d = get_object_pc_fp()
    object_pc_np = np.asarray(object_pc_o3d.points)
    object_normals_np = np.asarray(object_pc_o3d.normals)
    object_pc_com = object_pc_np.mean(axis=0)  # (3,)
    object_pc = torch.tensor(object_pc_np-object_pc_com, dtype=torch.float32, device=device).unsqueeze(0)
    # downsample
    object_pc, object_selected_indices = pytorch3d.ops.sample_farthest_points(object_pc, K=512)
    object_pc = object_pc.repeat(batch_size, 1, 1)
    object_normals = torch.tensor(object_normals_np[object_selected_indices.flatten().detach().cpu().numpy()], dtype=torch.float32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    t2 = time.time()
    print(f'********** SAM + FP: {t2 - t1:.2f} s')
    
    t1 = time.time()
    network = create_network(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"epoch_28.pth", map_location=device))
    network.eval()
    robot_name = "leaphand"
    hand = create_hand_model(robot_name, device=device)
    
    initial_q = hand.get_initial_q().to(device)
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
        transform, transform_pc = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)  #
        optim_transform = process_transform(hand.pk_chain, transform)

        layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(hand.pk_chain, layer, initial_q, optim_transform)  # (B, 22)
        outer_q, inner_q = controller(robot_name, predict_q)
    
    t2 = time.time()
    print(f'********** Model: {t2 - t1:.2f} s')

    
    t1 = time.time()
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
    # best_idx = np.argmin(each_loss_nmlzd)
    # choose one random in top 10
    best_idx = np.argsort(each_loss_nmlzd)[:5][np.random.randint(5)]
    print(f"best_idx: {best_idx}")
    
    best_q = predict_q[best_idx].detach().cpu().numpy()
    best_inner_q = inner_q[best_idx].detach().cpu().numpy()
    best_outer_q = outer_q[best_idx].detach().cpu().numpy()
    best_transform_pc = transform_pc[best_idx].detach().cpu().numpy()
    t2 = time.time()
    print(f'********** Filter: {t2 - t1:.2f} s')
    
    
    ''' vis and see whether execute '''
    sv = viser.ViserServer()
    best_q[:3] += object_pc_com
    best_inner_q[:3] += object_pc_com
    best_outer_q[:3] += object_pc_com
    best_transform_pc += object_pc_com
    hand_mesh = as_mesh(vis_hand.get_hand_trimesh(best_q, collision=True, visual=False)["collision"])
    hand_mesh_inner = as_mesh(vis_hand.get_hand_trimesh(best_inner_q, collision=True, visual=False)["collision"])
    hand_mesh_outer = as_mesh(vis_hand.get_hand_trimesh(best_outer_q, collision=True, visual=False)["collision"])

    hand_visual_mesh = vis_hand.get_hand_trimesh(best_q, collision=False, visual=True)["visual"]
    # get the grasp pose
    X_WorldLeapbase = vis_hand.current_status["base"].get_matrix().detach().cpu().numpy().reshape(4, 4)
    # first move to the pregrasp pose
    vec_object_center_to_hand_base = X_WorldLeapbase[:3, 3] - object_pc_com
    vec_object_center_to_hand_base = vec_object_center_to_hand_base / np.linalg.norm(vec_object_center_to_hand_base)
    X_WorldLeapbasepregrasp = X_WorldLeapbase.copy()
    X_WorldLeapbasepregrasp[:3, 3] = X_WorldLeapbase[:3, 3] + vec_object_center_to_hand_base * 0.04
    pregrasp_q = best_q.copy()
    pregrasp_q[:3] = X_WorldLeapbasepregrasp[:3, 3] 
    hand_mesh_pregrasp = as_mesh(vis_hand.get_hand_trimesh(pregrasp_q, collision=True, visual=False)["collision"])
    lgr.info(f"best_idx: {best_idx}")
    sv.scene.add_mesh_trimesh("hand_pregrasp", hand_mesh_pregrasp, visible=True)
    sv.scene.add_mesh_trimesh(f"hand_visual_{best_idx}", hand_visual_mesh)
    sv.scene.add_mesh_trimesh(f"hand_{best_idx}", hand_mesh, visible=False)
    sv.scene.add_mesh_trimesh(f"hand_inner_{best_idx}", hand_mesh_inner, visible=False)
    sv.scene.add_mesh_trimesh(f"hand_outer_{best_idx}", hand_mesh_outer, visible=False)
    # sv.scene.add_point_cloud("masked_pc_o3d_fusion", np.asarray(masked_pc_o3d_fusion.points), colors=np.asarray(masked_pc_o3d_fusion.colors), point_size=0.001, point_shape="circle")
    sv.scene.add_point_cloud("object_pc", np.asarray(object_pc_o3d.points), colors=(102, 192, 255), point_size=0.001, point_shape="circle")
    sv.scene.add_point_cloud("mlat_pc", best_transform_pc, colors=(255, 0, 0), point_size=0.001, point_shape="circle")
    
    validated = False
    validate_button = sv.gui.add_button("Execute",)
    # turn validated to True]
    def validate_true():
        nonlocal validated
        validated = True
        lgr.info("validated")
    validate_button.on_click(lambda _: validate_true())
    
    while True:
        time.sleep(0.1)
        lgr.info(f"validated: {validated}")
        if validated:
            break
    validated = False
    
    # if validated, execute
    lgr.info(f"Execute, best_idx: {best_idx}")
    xarm = XArm6RealWorld()
    leaphand = LeapNode()
    leaphand_control_time = 3.0
    
    planner_timestep = 1.0 / 20.0
    cmd_timestep = 1.0 / 100.0 
    
    t1 = time.time()
    ''' setup the planner and vis '''
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
    xarm6_planner.mplib_add_point_cloud(object_pc_np, name="object_pc")
    
    leapmount_data_dir = Path("third_party/xarm6/data/leapmount")
    leapmount_trimesh = trimesh.load_mesh(leapmount_data_dir / "hand_open_col_mesh_cvx.obj").apply_scale(1.1)
    leapmount_transform = np.load(leapmount_data_dir / "X_ArmLeapbase.npy")
    X_ArmLeapbase = leapmount_transform
    xarm6_planner.mplib_update_attached_object(
        leapmount_trimesh,
        leapmount_transform,
    )
    
    # change the floating hand pose to arm eef pose
    X_WorldEef = X_WorldLeapbase @ np.linalg.inv(X_ArmLeapbase)
    X_WorldEefpregrasp = X_WorldLeapbasepregrasp @ np.linalg.inv(X_ArmLeapbase)
    current_joint_values = np.array(xarm.get_joint_values())
    print(f"current_joint_values={current_joint_values}")
    print(f"X_WorldEefpregrasp={X_WorldEefpregrasp}")
    planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, X_WorldEefpregrasp)
    t2 = time.time()
    print(f'********** Planner 1: {t2 - t1:.2f} s')
    # status, grasp_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, X_WorldEefpregrasp, mask=mask)
    
    # motion planning and check the result
    mp_is_success = planning_result['status'] == 'Success'
    if not mp_is_success:
        lgr.info(f"Collision-free planning: Fail")
    else:
        # leaphand.go_to_a_pos_with_curr_limit(leaphand.open_pos, leaphand_control_time)
        leaphand.set_leap(leaphand.open_pos)
        lgr.info(f"Collision-free planning: Success")
        waypt_joint_values_np = planning_result['position']
        end_joint_values = waypt_joint_values_np[-1]
        update_viser_mp_result(sv, xarm6_pk, current_joint_values, end_joint_values, waypt_joint_values_np)

        while True:
            time.sleep(0.2)
            if validated:
                break
        validated = False
        
        # goto the pregrasp pose
        xarm.set_joint_values_sequence(waypt_joint_values_np, planning_timestep=planner_timestep)
        # leaphand.go_to_a_pos_during(leap_from_sim_to_rw(best_outer_q[6:], vis_hand.actuated_joint_names[6:]), 1.0)
        
        # arm and hand go to the grasp pose
        # current_joint_values = np.array(xarm.get_joint_values()).flatten()
        current_joint_values = np.array(xarm.get_joint_values())
        xarm6_planner.mplib_planner.remove_point_cloud("object_pc")
        xarm6_planner.mplib_planner.detach_object()
        xarm6_planner.mplib_planner.remove_point_cloud("env_pc")
        # breakpoint()
        print(f"current_joint_values={current_joint_values}")
        print(f"X_WorldEefpregrasp={X_WorldEefpregrasp}")
        print(f"X_WorldEef={X_WorldEef}")
        
        xarm6_planner = XARM6Planner(xarm6_planner_cfg)
        # planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, X_WorldEefpregrasp)
        # mask = [False, False, False, False, False, True]
        status, grasp_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, X_WorldEef)
        # planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, X_WorldEef)
        # assert False
        closest_grasp_arm_joint_values = get_closest_joint_value(current_joint_values, grasp_arm_joint_values)
        
        
        # breakpoint()
        #TODO
        # self implement the arm and the hand move at the same time 
        # the leaphand control rate is 4 times the xarm control rate
        mp_is_success = status == 'Success'
        
        if not mp_is_success:
            lgr.info(f"Grasp planning: Fail")
        else:
            xarm.arm.set_mode(6)
            xarm.arm.set_state(0)
            grasp_duration = 2.0
            
            waypt_joint_values_np = []
            waypt_joint_values_np.append(current_joint_values)
            waypt_joint_values_np.append(closest_grasp_arm_joint_values)
            
            # for i in range(len(grasp_arm_joint_values)):
            #     waypt_joint_values_np.append(grasp_arm_joint_values[i])
            waypt_joint_values_np = np.array(waypt_joint_values_np)
            
            
            # arm_joint_values_seq = min_jerk_interpolator_with_alpha(
            #     [current_joint_values, grasp_arm_joint_values],  grasp_duration, cmd_timestep
            # )
            arm_joint_values_seq = min_jerk_interpolator_with_alpha(
                waypt_joint_values_np,  grasp_duration, cmd_timestep
            )
            # calucate the grasp time according to the arm moving time
            mp_is_success = planning_result['status'] == 'Success'
            if not mp_is_success:
                lgr.info(f"Grasp planning: Fail")
            else:
                lgr.info(f"Grasp planning: Success")
                grasp_time = 4.0
                lgr.info(f"Grasp time: {grasp_time}")
                # current_hand_joint_values = leaphand.read_pos()
                desired_joint_values = leap_from_sim_to_rw(best_inner_q[6:], vis_hand.actuated_joint_names[6:])
                # hand_joint_values_seq = min_jerk_interpolator_with_alpha(
                #     np.array([current_hand_joint_values, desired_joint_values]), grasp_time, cmd_timestep / 4
                # )
                # # breakpoint()
                joint_values = arm_joint_values_seq[-1]
                xarm.arm.set_servo_angle(angle=xarm.to_list(joint_values), speed=0.1, wait=False, is_radian=True)
                
                # for cmd_idx, joint_values in enumerate(arm_joint_values_seq):
                #     # xarm.arm.set_servo_angle(angle=xarm.to_list(joint_values), speed=25, wait=False, is_radian=True)
                    
                #     for i in range(4):
                #         set_idx = cmd_idx * 4 + i
                #         if set_idx >= len(hand_joint_values_seq):
                #             break
                #         leaphand.set_leap(hand_joint_values_seq[cmd_idx * 4 + i])
                #         # leaphand.set_leap(hand_joint_values_seq[-1])
                #         time.sleep(cmd_timestep / 4)
                print(f"desired_joint_values={desired_joint_values}")
                
                # desired_joint_values = np.array(
                #                     [3.6968937, 3.109379, 4.4270687, 3.2060199,
                #                     3.7183695, 3.2382333, 4.3657093, 3.19835, 
                #                     1.9788352, 3.8502917, 2.072408, 3.0648937, 
                #                     5.641981, 4.3196898, 3.2581751, 4.6586995]
                #                 )
                # desired_joint_values =
                # [2.83000536 4.48018912 4.06303522 3.50779426
                #  2.73598369 4.23527113 4.00001261 3.65463725 
                #  1.1874038  4.33513466 2.53980054 4.57316104
                #  4.33577016 3.70832101 3.85468462 4.21806589]
                # desired_joint_values =
                # [3.6968937 3.109379  4.4270687 3.2060199
                #  3.7183695 3.2382333 4.3657093 3.19835   
                #  1.9788352 3.8502917 2.072408  3.0648937
                #  5.641981  4.3196898 3.2581751 4.6586995]
                # breakpoint()
                # print(f"desired_joint_values={desired_joint_values}")
                leaphand.pos_lim = 0.05
                leaphand.go_to_a_pos_with_curr_limit(desired_joint_values, grasp_time)
                # leaphand.set_leap(desired_joint_values)
                        
                xarm.arm.set_mode(0)
                xarm.arm.set_state(0)
                
                # lift the object by 10cm 
                current_joint_values = np.array(xarm.get_joint_values())
                X_WorldEeflift = X_WorldEef.copy()
                X_WorldEeflift[:3, 3] = X_WorldEef[:3, 3] + np.array([0, 0, 0.15])
                # breakpoint()
                
                status, lifted_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, X_WorldEeflift)
                mp_is_success = status == 'Success'
                
                closest_lifted_arm_joint_value = get_closest_joint_value(current_joint_values, lifted_arm_joint_values)
                
                
                if not mp_is_success:
                    lgr.info(f"Lift planning: Fail")
                else:
                    lgr.info(f"Lift planning: Success")
                    waypt_joint_values_np = planning_result['position']
                    lift_duration = 2
                    
                    waypt_joint_values_np = []
                    waypt_joint_values_np.append(current_joint_values)
                    waypt_joint_values_np.append(closest_lifted_arm_joint_value)
                    # for i in range(len(lifted_arm_joint_values)):
                    #     waypt_joint_values_np.append(lifted_arm_joint_values[i])
                    waypt_joint_values_np = np.array(waypt_joint_values_np)
                    
                    # xarm.set_joint_values_sequence([current_joint_values, lifted_arm_joint_values], lift_duration)
                    xarm.set_joint_values_sequence(waypt_joint_values_np, lift_duration)
                time.sleep(5)
                
                # leaphand.go_to_a_pos_with_curr_limit(leaphand.open_pos, leaphand_control_time)
                leaphand.set_leap(leaphand.open_pos)
                go_home_duration = 2
                waypt_joint_values_np = []
                current_joint_values = np.array(xarm.get_joint_values())
                waypt_joint_values_np.append(current_joint_values)
                waypt_joint_values_np.append(xarm.default_joint_values)
                waypt_joint_values_np = np.array(waypt_joint_values_np)
                xarm.set_joint_values_sequence(waypt_joint_values_np, go_home_duration)
                xarm.set_joint_values(waypt_joint_values_np[-1], speed=10, wait=True)
                
                num_break_down_motor, break_down_motor_list = count_break_down_motor(leaphand)
                print(f"num_break_down_motor = {num_break_down_motor}")
                print(f"break_down_motor_list = {break_down_motor_list}")
                
                
                
if __name__ == "__main__":
    main()
    

        