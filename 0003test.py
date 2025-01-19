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
from utils import PointCloudUtils

def get_canonical_pose_cam_from_file(step):
    file_path = f"/home/shaol/data/zjx/rw/matrix_list/{step}.npy"
    dicts = np.load(file_path, allow_pickle=True).item()
    base_pose_cam_canonical = dicts['SE3_base']
    target_pose_cam_canonical = dicts['SE3_tgt']
    base_name = dicts['mesh_name'][0]
    target_name = dicts['mesh_name'][1]
    return base_pose_cam_canonical, target_pose_cam_canonical, base_name, target_name

base_pose_cam_canonical, target_pose_cam_canonical, base_name, target_name = get_canonical_pose_cam_from_file(1)
sv = viser.ViserServer()
sv.scene.add_frame("target_pose", wxyz=R.from_matrix(target_pose_cam_canonical[:3,:3].T).as_quat()[[3, 0, 1, 2]], position=target_pose_cam_canonical[:3, 3], axes_length=0.3, axes_radius=0.01)

sv.scene.add_frame("base_pose", wxyz=R.from_matrix(base_pose_cam_canonical[:3,:3].T).as_quat()[[3, 0, 1, 2]], position=base_pose_cam_canonical[:3, 3], axes_length=0.3, axes_radius=0.01)
bp()