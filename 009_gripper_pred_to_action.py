import cv2
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
from xarm6_interface.utils.sam_prompt_drawer import SAMPromptDrawer

from xarm6_interface.arm_rw import XArm6RealWorld

from scipy.spatial.transform import Slerp
from xarm6_interface.arm_mplib import XARM6PlannerCfg, XARM6Planner, min_jerk_interpolator_with_alpha
import pytorch3d
from matplotlib import pyplot as plt
from loguru import logger as lgr
from leaphand_rw.leaphand_rw import LeapNode, leap_from_sim_to_rw
from xarm6_interface.envs.table_and_workspace_pc import WoodenTableMount, create_bounding_box_pc, create_plane_pc, env_pc_post_process
from xarm6_interface.arm_pk import XArm6WOEE
from xarm6_interface.utils.viser_utils import update_viser_mp_result
from third_party.FoundationPose.estimater import *
import pickle
from segment_anything import sam_model_registry, SamPredictor


def get_object_pc_fp(object_name):
    object_name_dino = object_name.replace("_", " ") + "."
    # Initialize the SAM predictor
    # sam = sam_model_registry[SAM_TYPE](checkpoint=SAM_PATH)
    # sam.to(device="cuda")
    # sam_predictor = SamPredictor(sam)
        
    
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
    arm_right_cam_X_BaseCamera_path = Path("third_party/xarm6/data/camera/mounted_white/1014_excalib_capture00/optimized_X_BaseCamera.npy")
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
        
        # input_boxes = grounding_dino_get_bbox(rgb, object_name_dino)
        # sam_predictor.set_image(rgb)
        
        # masks, scores, logits = sam_predictor.predict(
        #     point_coords=None,
        #     point_labels=None,
        #     box=input_boxes,
        #     multimask_output=False,
        # )
        # if masks.ndim == 4:
        #     masks = masks.squeeze(1)
        # mask_np = masks[0]
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
    
planner_timestep = 1.0 / 20.0
cmd_timestep = 1.0 / 100.0 
pregrasp_retreat_distance = 0.08
@hydra.main(version_base="1.2", config_path="", config_name="validate")
def main(cfg):
    sv = viser.ViserServer()

    # object_name = 'apple' # 4/5 ; 9/10 # ok
    # object_name = 'dinosaur'  #5/5 ； 9/10
    object_name = 'duck'  # 3/5 ; 8/10 # ok
    # object_name = 'flashlight' # ok
    # object_name = 'toilet_cleaner'  # 5/5 ; 10/10 # ok
    # object_name = 'rubic_cube'  #  4/5 ; 9/10 # ok
    # object_name = 'fish' # ok
    # object_name = 'gun' # ok
    # object_name = 'stage' # ok
    # object_name = 'iphone_box' # 2/5 ; 3/10 # ok
    # object_name = 'realsense_box' # 3/5 ; 5/10 # ok
    # object_name = 'brown_bottle' # ok
    
    batch_size = cfg.dataset.batch_size
    device = torch.device(f'cuda:{cfg.gpu}')
    
    # object_pc_o3d, masked_pc_o3d_fusion = get_object_pc()
    t1 = time.time()
    object_pc_o3d, X_WorldObject = get_object_pc_fp(object_name)
    object_grasp_pkl_path = Path("object_mesh_grasp") / object_name / "grasp.pkl"
    hand_open_mesh_path = Path("data/data_urdf/robot/xarm_gripper/hand_open.obj")
    hand_open_mesh = trimesh.load_mesh(hand_open_mesh_path)
    with open(object_grasp_pkl_path, "rb") as f:
        X_ObjectEE = pickle.load(f)  # (n, 4, 4)
    # X_WorldObject: shape=(4, 4)
    X_WorldEE = X_WorldObject[np.newaxis, ...] @ X_ObjectEE  
    X_WorldEE = filter_top_down_grasps(X_WorldEE, clip_min_z=0.18, approach_direction="z")
    
    
    vec_object_center_to_hand_base = - X_WorldEE[:, :3, 2]
    
    X_WorldEEpregrasp = X_WorldEE.copy()
    X_WorldEEpregrasp[:, :3, 3] = X_WorldEEpregrasp[:, :3, 3] + vec_object_center_to_hand_base * pregrasp_retreat_distance
    
    sv.scene.add_point_cloud("object_pc", points=np.asarray(object_pc_o3d.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
    for i in range(X_WorldEE.shape[0]):
        wxyz = trimesh.transformations.quaternion_from_matrix(X_WorldEE[i])
        pos = X_WorldEE[i][:3, 3]
        pre_wxyz = trimesh.transformations.quaternion_from_matrix(X_WorldEEpregrasp[i])
        pre_pos = X_WorldEEpregrasp[i][:3, 3]
        sv.scene.add_frame(f"object_ee_{i}", wxyz=wxyz, position=pos, axes_length=0.03, axes_radius=0.001, visible=False)
        # sv.scene.add_frame(f"object_ee_pregrasp_{i}", wxyz=pre_wxyz, position=pre_pos, axes_length=0.03, axes_radius=0.001)
        
        
    ''' setup the planner and vis '''
    object_pc_np = np.asarray(object_pc_o3d.points)
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
    
    grippermount_data_dir = Path("data/data_urdf/robot/xarm_gripper/hand_open_cvx_hull.obj")
    gripper_trimesh = trimesh.load_mesh(grippermount_data_dir).apply_scale(1.1)
    gripper_transform = np.eye(4)
    xarm6_planner.mplib_update_attached_object(
        gripper_trimesh,
        gripper_transform,
    )
    
    xarm = XArm6RealWorld()
    xarm.arm.set_gripper_position(850, wait=True)
    
    current_joint_values = np.array(xarm.get_joint_values())
    valid_pregrasp_idx = 0
    # generate a random list of int length X_WorldEEpregrasp.shape[0], but the idx order is random
    random_idx_list = list(range(X_WorldEEpregrasp.shape[0]))
    np.random.shuffle(random_idx_list)
    for i in random_idx_list:
        planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, X_WorldEEpregrasp[i])
        if planning_result['status'] == 'Success':
            valid_pregrasp_idx = i
            break
    if planning_result['status'] != 'Success':
        lgr.info(f"Collision-free planning: Fail")
        return
    selected_X_WorldEE = X_WorldEE[valid_pregrasp_idx]
    wxyz = trimesh.transformations.quaternion_from_matrix(selected_X_WorldEE)
    pos = selected_X_WorldEE[:3, 3]
    sv.scene.add_mesh_simple(f"hand_open_{i}", vertices=hand_open_mesh.vertices, faces=hand_open_mesh.faces, wxyz=wxyz, position=pos, opacity=0.5)
    
    lgr.info(f"Collision-free planning: Success")
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
    xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.35, wait=True)
    
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    current_joint_values = np.array(xarm.get_joint_values()) # yiwen
    status, grasp_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, X_WorldEE[valid_pregrasp_idx])
    closest_grasp_arm_joint_values = get_closest_joint_value(current_joint_values, grasp_arm_joint_values)
    
    # waypoints
    # planning_result = xarm6_planner.mplib_plan_pose(current_joint_values, X_WorldEE[valid_pregrasp_idx])
    
    mp_is_success = status == 'Success'
    if not mp_is_success:
        lgr.info(f"Grasp planning: Fail")
    else:
        grasp_joint_values = closest_grasp_arm_joint_values
        xarm.arm.set_servo_angle(angle=xarm.to_list(grasp_joint_values), speed=0.2, wait=True, is_radian=True)
        xarm.arm.set_gripper_position(-10, wait=True)
        
        current_joint_values = np.array(xarm.get_joint_values())
        X_WorldEeflift =  X_WorldEE[valid_pregrasp_idx].copy()
        X_WorldEeflift[:3, 3] =  X_WorldEE[valid_pregrasp_idx][:3, 3] + np.array([0, 0, 0.15])
        status, lifted_arm_joint_values = xarm6_planner.mplib_ik(current_joint_values, X_WorldEeflift)
        mp_is_success = status == 'Success'
        closest_lifted_arm_joint_value = get_closest_joint_value(current_joint_values, lifted_arm_joint_values)
        if not mp_is_success:
            lgr.info(f"Lift planning: Fail")
        else:
            lgr.info(f"Lift planning: Success")
            xarm.arm.set_servo_angle(angle=xarm.to_list(closest_lifted_arm_joint_value), speed=0.2, wait=True, is_radian=True)
            time.sleep(3)
            xarm.arm.set_gripper_position(850, wait=True)
            
            go_home_duration = 2
            waypt_joint_values_np = []
            current_joint_values = np.array(xarm.get_joint_values())
            waypt_joint_values_np.append(current_joint_values)
            waypt_joint_values_np.append(xarm.default_joint_values)
            waypt_joint_values_np = np.array(waypt_joint_values_np)
            xarm.set_joint_values_sequence(waypt_joint_values_np, go_home_duration)
            xarm.set_joint_values(waypt_joint_values_np[-1], speed=0.2, wait=True)
            
    

    
    

    
    



                  
if __name__ == "__main__":
    main()
    

        