from argparse import ArgumentParser
import cv2
import time

import trimesh
import viser
import numpy as np
import open3d as o3d
import hydra
import torch
import warnings
from xarm6_interface import XARM6_IP, XARM6LEFT_IP
from xarm.wrapper import XArmAPI

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR + "/third_party/segment-anything"))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print(sys.path)
PARENT_DIR = os.path.abspath(os.path.join(ROOT_DIR, '..'))
sys.path.append(PARENT_DIR)
from ..BiMo.code import models, utils
from xarm6_interface.utils.realsense import MultiRealsense, get_masked_pointcloud, remove_outliers
from pathlib import Path
from scipy.spatial.transform import Rotation as R

from xarm6_interface import SAM_TYPE, SAM_PATH
from xarm6_interface.utils.sam_prompt_drawer import SAMPromptDrawer
from xarm6_interface.utils.gpis import gpis_fit
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
from sklearn.decomposition import PCA

parser = ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--categories',
                    type=str,
                    help='list all categories [Default: None, meaning all 10 categories]',
                    default=None)
parser.add_argument('--primact_type', type=str)
parser.add_argument('--aff1_version', type=str, default=None)
parser.add_argument('--aff1_path', type=str, default=None)
parser.add_argument('--aff1_eval_epoch', type=str, default=None)
parser.add_argument('--actor1_version', type=str, default=None, help='model def file')
parser.add_argument('--actor1_path', type=str)
parser.add_argument('--actor1_eval_epoch', type=str)
parser.add_argument('--critic1_version', type=str, default=None, help='model def file')
parser.add_argument('--critic1_path', type=str)
parser.add_argument('--critic1_eval_epoch', type=str)

parser.add_argument('--aff2_version', type=str, default=None)
parser.add_argument('--aff2_path', type=str, default=None)
parser.add_argument('--aff2_eval_epoch', type=str, default=None)
parser.add_argument('--actor2_version', type=str, default=None, help='model def file')
parser.add_argument('--actor2_path', type=str)
parser.add_argument('--actor2_eval_epoch', type=str)
parser.add_argument('--critic2_version', type=str, default=None, help='model def file')
parser.add_argument('--critic2_path', type=str)
parser.add_argument('--critic2_eval_epoch', type=str)

parser.add_argument('--use_CA', action='store_true', default=False)
parser.add_argument('--CA_path', type=str)
parser.add_argument('--CA_eval_epoch', type=str)
args = parser.parse_args()


def get_object_pc(object_name='Box'):
    prompt_drawer = SAMPromptDrawer(window_name="Prompt Drawer",
                                    screen_scale=2.0,
                                    sam_checkpoint=SAM_PATH,
                                    device="cuda",
                                    model_type=SAM_TYPE)
    ''' multi realsense setup '''
    arm_left_cam_serial = "147122075879"
    arm_right_cam_serial = "241122074374"
    camera_serial_nums = [arm_left_cam_serial, arm_right_cam_serial]
    multi_rs = MultiRealsense(camera_serial_nums)
    arm_left_cam_K_path = Path("third_party/xarm6/data/camera/mounted_black/K.npy")
    arm_left_cam_X_BaseCamera_path = Path(
        "third_party/xarm6/data/camera/mounted_black/0914_excalib_capture1/optimized_X_BaseCamera.npy")
    arm_left_cam_K = np.load(arm_left_cam_K_path)
    arm_left_cam_X_BaseCamera = np.load(arm_left_cam_X_BaseCamera_path)

    arm_right_cam_K_path = Path("third_party/xarm6/data/camera/mounted_white/K.npy")
    arm_right_cam_X_BaseCamera_path = Path(
        "third_party/xarm6/data/camera/mounted_white/0914_excalib_capture3/optimized_X_BaseCamera.npy")
    arm_right_cam_K = np.load(arm_right_cam_K_path)
    arm_right_cam_X_BaseCamera = np.load(arm_right_cam_X_BaseCamera_path)

    multi_rs.set_intrinsics(0, arm_left_cam_K[0, 0], arm_left_cam_K[1, 1], arm_left_cam_K[0, 2], arm_left_cam_K[1, 2])
    multi_rs.set_intrinsics(1, arm_right_cam_K[0, 0], arm_right_cam_K[1, 1], arm_right_cam_K[0, 2], arm_right_cam_K[1,
                                                                                                                    2])
    camera_wxyzs = [
        R.from_matrix(arm_left_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
        R.from_matrix(arm_right_cam_X_BaseCamera[:3, :3]).as_quat()[[3, 0, 1, 2]],
    ]
    camera_positions = [arm_left_cam_X_BaseCamera[:3, 3], arm_right_cam_X_BaseCamera[:3, 3]]
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
        masked_pc_o3d = get_masked_pointcloud(rgb, depth, shrink_mask_np.reshape(h, w),
                                              multi_rs.camera_data[camera_idx]["pinhole_camera_intrinsic"])
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

    return fitted_pcd, masked_pc_o3d_fusion, rgb


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


def get_pcd_center(pcd):
    return np.mean(np.asarray(pcd.points), axis=0)


planner_timestep = 1.0 / 50.0
cmd_timestep = 1.0 / 100.0
pregrasp_retreat_distance = 0.08
# @hydra.main(version_base="1.2", config_path="", config_name="validate")


def get_predictions(pcs, ctpt1_list, ctpt2_list):
    # load models
    aff1_def = utils.get_model_module(args.aff1_version)
    affordance1 = aff1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)
    actor1_def = utils.get_model_module(args.actor1_version)
    actor1 = actor1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim)
    critic1_def = utils.get_model_module(args.critic1_version)
    critic1 = critic1_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    aff2_def = utils.get_model_module(args.aff2_version)
    affordance2 = aff2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)
    actor2_def = utils.get_model_module(args.actor2_version)
    actor2 = actor2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, z_dim=args.z_dim)
    critic2_def = utils.get_model_module(args.critic2_version)
    critic2 = critic2_def.Network(args.feat_dim, args.cp_feat_dim, args.dir_feat_dim, task_input_dim=task_input_dim)

    if not args.use_CA:
        affordance1.load_state_dict(
            torch.load(os.path.join(args.aff1_path, 'ckpts', '%s-network.pth' % args.aff1_eval_epoch)))
        actor1.load_state_dict(
            torch.load(os.path.join(args.actor1_path, 'ckpts', '%s-network.pth' % args.actor1_eval_epoch)))
        critic1.load_state_dict(
            torch.load(os.path.join(args.critic1_path, 'ckpts', '%s-network.pth' % args.critic1_eval_epoch)))
        affordance2.load_state_dict(
            torch.load(os.path.join(args.aff2_path, 'ckpts', '%s-network.pth' % args.aff2_eval_epoch)))
        actor2.load_state_dict(
            torch.load(os.path.join(args.actor2_path, 'ckpts', '%s-network.pth' % args.actor2_eval_epoch)))
        critic2.load_state_dict(
            torch.load(os.path.join(args.critic2_path, 'ckpts', '%s-network.pth' % args.critic2_eval_epoch)))
    else:
        affordance1.load_state_dict(
            torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance1.pth' % args.CA_eval_epoch)))
        actor1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor1.pth' % args.CA_eval_epoch)))
        critic1.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic1.pth' % args.CA_eval_epoch)))
        affordance2.load_state_dict(
            torch.load(os.path.join(args.CA_path, 'ckpts', '%s-affordance2.pth' % args.CA_eval_epoch)))
        actor2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-actor2.pth' % args.CA_eval_epoch)))
        critic2.load_state_dict(torch.load(os.path.join(args.CA_path, 'ckpts', '%s-critic2.pth' % args.CA_eval_epoch)))

    affordance1.to(device).eval()
    actor1.to(device).eval()
    critic1.to(device).eval()
    affordance2.to(device).eval()
    actor2.to(device).eval()
    critic2.to(device).eval()

    raw_pair = len(ctpt1_list)
    ctpt1 = torch.tensor(np.array(ctpt1_list)).float().reshape(batch_size * raw_pair, -1).to(args.device)
    ctpt2 = torch.tensor(np.array(ctpt2_list)).float().reshape(batch_size * raw_pair, -1).to(args.device)

    num_ctpt1, num_ctpt2, rv1, rv2 = raw_pair, raw_pair, args.rv1, args.rv2
    num_pair1 = args.num_pair1
    batch_size = 1

    #inference
    with torch.no_grad():  #TODO
        #aff1
        aff_scores = affordance1.forward_n(pcs, ctpt1, raw_pair).view(batch_size, raw_pair)  # B * N

        aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, raw_pair)

        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)

        num_ctpt1 = min(num_ctpt1, max(1, int(raw_pair * args.aff1_topk)))
        selected_idx_idx = torch.randint(0, max(1, int(raw_pair * args.aff1_topk)), size=(batch_size, num_ctpt1))
        selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]

        position1s = ctpt1.reshape(batch_size, -1, 3)[batch_idx, selected_idx].view(batch_size * num_ctpt1, 3)
        position2s = ctpt2.reshape(batch_size, -1, 3)[batch_idx, selected_idx].view(batch_size * num_ctpt1, 3)

        #actor1
        dir1s = actor1.actor_sample_n_finetune(pcs, position1s, rvs_ctpt=num_ctpt1,
                                               rvs=rv1).contiguous().view(batch_size * num_ctpt1 * rv1, 6)
        #critic1
        critic_scores = critic1.forward_n_finetune(pcs, position1s, dir1s, rvs_ctpt=num_ctpt1,
                                                   rvs=rv1).view(batch_size, num_ctpt1 * rv1)
        critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt1 * rv1)
        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
        selected_idx_idx = torch.randint(0, int(num_ctpt1 * rv1 * args.critic_topk1), size=(batch_size, num_pair1))
        selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]

        position1 = position1s.view(batch_size, num_ctpt1, 3)[batch_idx,
                                                              selected_idx // rv1].view(batch_size * num_pair1, 3)
        position2 = position2s.view(batch_size, num_ctpt1, 3)[batch_idx,
                                                              selected_idx // rv1].view(batch_size * num_pair1, 3)

        dir1 = dir1s.view(batch_size, num_ctpt1 * rv1, 6)[batch_idx, selected_idx].view(batch_size * num_pair1, 6)

        # print('aff2') #TODO
        aff_scores = affordance2.forward_n(
            pcs,
            position1,
            position2,
            dir1,
        ).view(batch_size, num_pair1)
        aff_sorted_idx = torch.argsort(aff_scores, dim=1, descending=True).view(batch_size, num_pair1)
        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)

        num_ctpt2 = min(num_ctpt2, max(1, int(num_pair1 * args.aff2_topk)))
        selected_idx_idx = torch.randint(0, max(1, int(num_pair1 * args.aff2_topk)), size=(batch_size, num_ctpt2))
        selected_idx = aff_sorted_idx[batch_idx, selected_idx_idx]

        position2 = position2.reshape(batch_size, -1, 3)[batch_idx, selected_idx].view(batch_size * num_ctpt2, 3)
        position1 = position1.reshape(batch_size, -1, 3)[batch_idx, selected_idx].view(batch_size * num_ctpt2, 3)
        dir1 = dir1.reshape(batch_size, -1, 6)[batch_idx, selected_idx].view(batch_size * num_ctpt2, 6)
        #actor2
        dir2s = actor2.actor_sample_n_finetune(pcs, position1, position2, dir1, rvs_ctpt=num_ctpt2,
                                               rvs=rv2).contiguous().view(batch_size * num_ctpt2 * rv2, 6)

        # print('critic2')
        expanded_dir1s = dir1.unsqueeze(dim=1).repeat(1, rv2, 1).reshape(batch_size * num_ctpt2 * rv2, 6)
        critic_scores = critic2.forward_n_finetune(pcs,
                                                   position1,
                                                   position2,
                                                   expanded_dir1s,
                                                   dir2s,
                                                   rvs_ctpt=num_ctpt2,
                                                   rvs=rv2).view(batch_size, num_ctpt2 * rv2)
        critic_sorted_idx = torch.argsort(critic_scores, dim=1, descending=True).view(batch_size, num_ctpt2 * rv2)
        batch_idx = torch.tensor(range(batch_size)).view(batch_size, 1)
        selected_idx_idx = torch.randint(0, max(1, int(num_ctpt2 * rv2 * args.critic_topk)), size=(batch_size, 1))
        selected_idx = critic_sorted_idx[batch_idx, selected_idx_idx]
        pred_scores = critic_scores[batch_idx, selected_idx]

        position1 = position1.reshape(batch_size, num_ctpt2, 3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
        position2 = position2.reshape(batch_size, num_ctpt2, 3)[batch_idx, selected_idx // rv2].view(batch_size, 3)
        dir1 = expanded_dir1s.view(batch_size, num_ctpt2 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)
        dir2 = dir2s.view(batch_size, num_ctpt2 * rv2, 6)[batch_idx, selected_idx].view(batch_size, 6)

    with torch.no_grad():
        if args.draw_aff_map and repeat_i < args.num_draw:
            aff_scores = affordance1.inference_whole_pc(pcs).detach().cpu().numpy().reshape(-1)
            aff_scores = 1 / (1 + np.exp(-(aff_scores - 0.5) * 15))
            fn = os.path.join(save_aff_dir, '%d_%s_%s_%s_%s' % (file_id, str(repeat_i), category, shape_id, 'map1'))
            utils.draw_affordance_map(fn,
                                      cam2cambase,
                                      mat44,
                                      pcs[0].detach().cpu().numpy(),
                                      aff_scores,
                                      coordinate_system=args.coordinate_system,
                                      ctpt1=position1[0].detach().cpu().numpy(),
                                      type='0')
            aff_scores = affordance2.inference_whole_pc(pcs, position1,
                                                        dir1).detach().cpu().numpy().reshape(-1)  # B * N * 1
            aff_scores = 1 / (1 + np.exp(-(aff_scores - 0.5) * 15))
            fn = os.path.join(save_aff_dir, '%d_%s_%s_%s_%s' % (file_id, str(repeat_i), category, shape_id, 'map2'))
            utils.draw_affordance_map(fn,
                                      cam2cambase,
                                      mat44,
                                      pcs[0].detach().cpu().numpy(),
                                      aff_scores,
                                      coordinate_system=args.coordinate_system,
                                      ctpt1=position1[0].detach().cpu().numpy(),
                                      ctpt2=position2[0].detach().cpu().numpy(),
                                      type='2')

        dir1 = dir1.view(6).detach().cpu().numpy()
        dir2 = dir2.view(6).detach().cpu().numpy()

        position1 = position1.view(3).detach().cpu().numpy()
        position2 = position2.view(3).detach().cpu().numpy()
        up1, forward1 = dir1[0:3], dir1[3:6]
        up2, forward2 = dir2[0:3], dir2[3:6]


def get_ctpts(src_img_PIL, trg_img_PIL):
    pass


def cal_rotmat(CAM_In_mat,
               position_world,
               up,
               forward,
               number,
               out_info,
               start_dist,
               displacement,
               maneuver_dist=0,
               is_given_action=False,
               action_direction_world=np.array([1, 0, 0])):
    up /= np.linalg.norm(up)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)

    forward_cam = np.linalg.inv(CAM_In_mat) @ forward
    up_cam = np.linalg.inv(CAM_In_mat) @ up

    if is_given_action:
        action_direction_world /= np.linalg.norm(action_direction_world)
    else:
        action_direction_world = left  #Plier

    rotmat = np.eye(4).astype(np.float32)  # rotmat: world coordinate
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - up * start_dist
    # start_pose = Pose(start_rotmat) #TODO
    start_pose = start_rotmat

    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = start_rotmat[:3, 3] + action_direction_world * displacement
    # final_pose = Pose(final_rotmat)
    final_pose = final_rotmat

    pre_rotmat = np.array(rotmat, dtype=np.float32)
    pre_rotmat[:3, 3] = start_rotmat[:3, 3] - action_direction_world * maneuver_dist
    pre_pose = pre_rotmat
    if out_info is not None:
        out_info['position_world' + number] = position_world.tolist()  # world
        out_info['gripper_direction_world' + number] = up.tolist()
        out_info['gripper_direction_camera' + number] = up_cam.tolist()
        out_info['gripper_forward_direction_world' + number] = forward.tolist()
        out_info['gripper_forward_direction_camera' + number] = forward_cam.tolist()
        out_info['action_direction_world' + number] = action_direction_world.tolist()
        out_info['pre_rotmat_world' + number] = pre_rotmat.tolist()
        out_info['start_rotmat_world' + number] = start_rotmat.tolist()
        out_info['final_rotmat_world' + number] = final_rotmat.tolist()
        out_info['start_dist' + number] = float(start_dist)
        out_info['displacement' + number] = float(displacement)
        out_info['maneuver_dist' + number] = float(maneuver_dist)

    return pre_pose, pre_rotmat, start_pose, start_rotmat, final_pose, final_rotmat


def retreive_pipline(trg_rgb):
    pass


def init_dual_arm(ip_l="192.168.1.232", ip_r="192.168.1.208"):
    arm_l = XArmAPI(ip_l)
    arm_l.motion_enable(enable=True)
    arm_l.set_mode(0)
    arm_l.set_state(state=0)
    arm_r = XArmAPI(ip_r)
    arm_r.motion_enable(enable=True)
    arm_r.set_mode(0)
    arm_r.set_state(state=0)

    # init gripper
    arm_l.set_gripper_mode(0)
    arm_r.set_gripper_mode(0)

    arm_l.set_gripper_enable(True)
    arm_r.set_gripper_enable(True)

    arm_l.set_gripper_speed(5000)
    arm_r.set_gripper_speed(5000)

    arm_l.set_gripper_position(450, wait=False)
    arm_r.set_gripper_position(450, wait=True)

    # set cartesian velocity control mode
    arm_l.set_mode(0)
    arm_l.set_state(state=0)
    arm_r.set_mode(0)
    arm_r.set_state(state=0)

    # init position
    arm_l.set_position(x=200, y=0, z=350, roll=-180, pitch=0, yaw=0, speed=100, wait=False)
    arm_r.set_position(x=200, y=0, z=350, roll=-180, pitch=0, yaw=0, speed=100, wait=True)
    time.sleep(1)

    return arm_l, arm_r


def unfolding(object_name='Box', arm1_ip=XARM6LEFT_IP, arm2_ip=XARM6_IP):

    t1 = time.time()
    ''' setup the planner and vis '''
    # sv = viser.ViserServer()
    xarm6_pk = XArm6WOEE()
    xarm6_planner_cfg = XARM6PlannerCfg(vis=False, n_env_pc=10000, timestep=planner_timestep)
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    env_params = WoodenTableMount()
    workspace_pc = create_bounding_box_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax,
                                          env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    table_plane_pc = create_plane_pc(env_params.table_plane_xmin, env_params.table_plane_ymin,
                                     env_params.table_plane_zmin, env_params.table_plane_xmax,
                                     env_params.table_plane_ymax, env_params.table_plane_zmax,
                                     xarm6_planner_cfg.n_env_pc)
    workspace_xmin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmin,
                                        env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymin_pc = create_plane_pc(env_params.xmin, env_params.ymin, env_params.zmin, env_params.xmax,
                                        env_params.ymin, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    workspace_ymax_pc = create_plane_pc(env_params.xmin, env_params.ymax, env_params.zmin, env_params.xmax,
                                        env_params.ymax, env_params.zmax, xarm6_planner_cfg.n_env_pc)
    env_pc = np.concatenate([workspace_pc, table_plane_pc, workspace_xmin_pc, workspace_ymin_pc, workspace_ymax_pc],
                            axis=0)
    env_pc = env_pc_post_process(env_pc, filter_norm_thresh=0.1, n_save_pc=None)
    xarm6_planner.mplib_add_point_cloud(env_pc, name="env_pc")
    ''' setup the planner and vis '''
    '''init dual arm'''

    print('Initial dual arms')
    xarm1, xarm2 = init_dual_arm()
    ''' get the object pc and rgb'''
    #get obj pc
    object_pc_o3d, masked_pc_o3d_fusion, rgb = get_object_pc()
    object_pc_np = np.asarray(object_pc_o3d.points)
    object_normals_np = np.asarray(object_pc_o3d.normals)
    object_pc_com = object_pc_np.mean(axis=0)  # (3,)
    object_pc = torch.tensor(object_pc_np - object_pc_com, dtype=torch.float32, device=device).unsqueeze(0)
    # downsample
    object_pc, object_selected_indices = pytorch3d.ops.sample_farthest_points(object_pc, K=512)
    object_pc = object_pc.repeat(batch_size, 1, 1)
    object_normals = torch.tensor(object_normals_np[object_selected_indices.flatten().detach().cpu().numpy()],
                                  dtype=torch.float32,
                                  device=device).unsqueeze(0).repeat(batch_size, 1, 1)
    canonicalized_pcd, rotation_matrice, centroid = canonicalize_point_cloud(object_pc_o3d)
    ''' get the object pc and rgb'''
    '''retrieve and correspond'''
    trg_img_PIL = rgb
    src_img_np = retreive_pipline(trg_img_PIL)  #TODO
    ctpt1_list, ctpt2_list = correspond_pipline(src_img_np)  #TODO
    '''retrieve and correspond'''
    '''inference'''
    position1, up1, forward1, position2, up2, forward2 = get_predictions(object_pc, ctpt1_list, ctpt2_list)  #TODO
    '''inference'''
    action_dir1, action_dir2 = utils.get_action_dir(args.primact_type, up1, forward1, up2, forward2)
    '''cal pose'''
    pre_pose1, pre_rotmat1, start_pose1, start_rotmat1, final_pose1, final_rotmat1 = cal_rotmat(
        CAM_IN_MAT,  #TODO
        position1,
        up1,
        forward1,
        number='1',
        start_dist=start_dist1,
        displacement=displacement1,
        maneuver_dist=maneuver_dist1,
        is_given_action=True,
        action_direction_world=action_dir1)
    pre_pose2, pre_rotmat2, start_pose2, start_rotmat2, final_pose2, final_rotmat2 = cal_rotmat(
        CAM_IN_MAT,  #TODO
        position2,
        up2,
        forward2,
        number='2',
        start_dist=start_dist2,
        displacement=displacement2,
        maneuver_dist=maneuver_dist2,
        is_given_action=True,
        action_direction_world=action_dir2)
    '''cal pose'''

    sv.scene.add_point_cloud("canonical_pc",
                             points=np.asarray(canonicalized_pcd.points),
                             colors=(0, 255, 0),
                             point_size=0.002,
                             point_shape="circle")
    sv.scene.add_frame("canonical_pose",
                       wxyz=R.from_matrix(rotation_matrice[:3, :3]).as_quat()[[3, 0, 1, 2]],
                       position=centroid,
                       axes_length=0.03,
                       axes_radius=0.001)
    pcd_center = get_pcd_center(object_pc_o3d)
    sv.scene.add_point_cloud("object_pc",
                             points=np.asarray(object_pc_o3d.points),
                             colors=(255, 0, 0),
                             point_size=0.002,
                             point_shape="circle")

    # REAL PALNNING IS HERE!
    current_joint_values1 = np.array(xarm1.get_joint_values())
    planning_result1 = xarm6_planner.mplib_plan_pose(current_joint_values1, pre_pose1)
    if planning_result1['status'] != 'Success':
        lgr.info(f"Collision-free planning: Fail")
        return
    lgr.info(f"Collision-free planning 1: Success")

    waypt_joint_values_np1 = planning_result1['position']
    end_joint_values1 = waypt_joint_values_np1[-1]
    update_viser_mp_result(sv, xarm6_pk, current_joint_values1, end_joint_values1, waypt_joint_values_np1)

    current_joint_values2 = np.array(xarm2.get_joint_values())
    planning_result2 = xarm6_planner.mplib_plan_pose(current_joint_values2, pre_pose2)
    if planning_result2['status'] != 'Success':
        lgr.info(f"Collision-free planning: Fail")
        return

    lgr.info(f"Collision-free planning 2: Success")

    waypt_joint_values_np2 = planning_result2['position']
    end_joint_values2 = waypt_joint_values_np2[-1]
    update_viser_mp_result(sv, xarm6_pk, current_joint_values2, end_joint_values2, waypt_joint_values_np2)

    xarm1.set_joint_values_sequence(waypt_joint_values_np1, planning_timestep=planner_timestep)
    xarm1.set_joint_values(waypt_joint_values_np1[-1], speed=0.35, wait=True)

    input("Press Enter to continue...")
    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    current_joint_values1 = np.array(xarm1.get_joint_values())  # yiwen
    status1, grasp_arm_joint_values1 = xarm6_planner.mplib_ik(current_joint_values1, pre_pose1)
    closest_grasp_arm_joint_values1 = get_closest_joint_value(current_joint_values1, grasp_arm_joint_values1)
    mp_is_success1 = status1 == 'Success'

    xarm6_planner = XARM6Planner(xarm6_planner_cfg)
    current_joint_values2 = np.array(xarm2.get_joint_values())  # yiwen
    status2, grasp_arm_joint_values2 = xarm6_planner.mplib_ik(current_joint_values2, pre_pose2)
    closest_grasp_arm_joint_values2 = get_closest_joint_value(current_joint_values2, grasp_arm_joint_values2)
    mp_is_success2 = status2 == 'Success'

    mp_is_success = (mp_is_success1 and mp_is_success2)
    if not mp_is_success:
        lgr.info(f"Grasp planning: Fail")
    else:
        grasp_joint_values1 = closest_grasp_arm_joint_values1
        grasp_joint_values2 = closest_grasp_arm_joint_values2

        xarm1.arm.set_servo_angle(angle=xarm1.to_list(grasp_joint_values1), speed=0.2, wait=True, is_radian=True)
        xarm2.arm.set_servo_angle(angle=xarm2.to_list(grasp_joint_values2), speed=0.2, wait=True, is_radian=True)

        xarm1.arm.set_gripper_position(-10, wait=True)
        xarm2.arm.set_gripper_position(-10, wait=True)
        '''grasping success, lift it up'''
        current_joint_values1 = np.array(xarm1.get_joint_values())
        status1, lifted_arm_joint_values1 = xarm6_planner.mplib_ik(current_joint_values1, final_pose1)
        mp_is_success1 = status1 == 'Success'
        closest_lifted_arm_joint_value1 = get_closest_joint_value(current_joint_values1, lifted_arm_joint_values1)

        current_joint_values2 = np.array(xarm2.get_joint_values())
        status2, lifted_arm_joint_values2 = xarm6_planner.mplib_ik(current_joint_values2, final_pose2)
        mp_is_success2 = status2 == 'Success'
        closest_lifted_arm_joint_value2 = get_closest_joint_value(current_joint_values2, lifted_arm_joint_values2)

        mp_is_success = (mp_is_success1 and mp_is_success2)
        if not mp_is_success:
            lgr.info(f"Lift planning: Fail")
        else:
            lgr.info(f"Lift planning: Success")
            xarm1.arm.set_servo_angle(angle=xarm1.to_list(closest_lifted_arm_joint_value1),
                                      speed=0.2,
                                      wait=True,
                                      is_radian=True)

            xarm2.arm.set_servo_angle(angle=xarm2.to_list(closest_lifted_arm_joint_value2),
                                      speed=0.2,
                                      wait=True,
                                      is_radian=True)
            time.sleep(2)

            # xarm.arm.set_gripper_position(850, wait=True)

            input("Press Enter to continue...")

            # obstacle_pcd, pose = get_object_pc_fp(object_name='chair2')
            # print(f"current pose: {pose}")
            # sv.scene.add_frame("obstacle_pose", wxyz=R.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]], position=pose[:3, 3], axes_length=0.03, axes_radius=0.001)
            # sv.scene.add_point_cloud("object_pc", points=np.asarray(obstacle_pcd.points), colors=(255, 0, 0), point_size=0.002, point_shape="circle")
            # xarm6_planner.mplib_add_point_cloud(np.asarray(obstacle_pcd.points), name="obstacle_pc")

            go_home_duration = 2
            waypt_joint_values_np1 = []
            waypt_joint_values_np2 = []
            current_joint_values1 = np.array(xarm1.get_joint_values())
            current_joint_values2 = np.array(xarm2.get_joint_values())
            waypt_joint_values_np1.append(current_joint_values1)
            waypt_joint_values_np1.append(xarm1.default_joint_values)
            waypt_joint_values_np1 = np.array(waypt_joint_values_np1)

            waypt_joint_values_np2.append(current_joint_values2)
            waypt_joint_values_np2.append(xarm2.default_joint_values)
            waypt_joint_values_np2 = np.array(waypt_joint_values_np2)

            xarm1.set_joint_values_sequence(waypt_joint_values_np1, go_home_duration)
            xarm2.set_joint_values_sequence(waypt_joint_values_np2, go_home_duration)
            xarm1.set_joint_values(waypt_joint_values_np1[-1], speed=0.2, wait=True)
            xarm2.set_joint_values(waypt_joint_values_np2[-1], speed=0.2, wait=True)

            # get_current_attach_pose_offset(object_name='sticker')


if __name__ == "__main__":
    # from hydra.core.global_hydra import GlobalHydra
    # GlobalHydra.instance().clear()  # 清除全局状态
    # Global variables
    initial = True
    collision_pcd = None
    sv = viser.ViserServer()

    batch_size = 1  #TODO
    device = args.device

    unfolding()
