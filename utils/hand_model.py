import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import json
import math
import random
import torch
import trimesh
import pytorch_kinematics as pk

from utils.farthest_point_sampling import farthest_point_sampling
from utils.mesh_utils import load_link_geometries
from utils.rotation import *


class HandModel:
    def __init__(self,
                 robot_name,
                 urdf_path,
                 meshes_path,
                 links_pc_path,
                 device,
                 link_num_points=512):
        self.robot_name = robot_name
        self.urdf_path = urdf_path
        self.meshes_path = meshes_path
        self.device = device

        self.pk_chain = pk.build_chain_from_urdf(open(urdf_path).read()).to(dtype=torch.float32, device=device)
        self.dof = len(self.pk_chain.get_joint_parameter_names())
        if os.path.exists(links_pc_path):  # In case of generating robot links pc, the file doesn't exist.
            links_pc_data = torch.load(links_pc_path, map_location=device)
            self.links_pc = links_pc_data['filtered']
            self.links_pc_original = links_pc_data['original']
        else:
            self.links_pc = None
            self.links_pc_original = None

        self.meshes = load_link_geometries(robot_name, self.urdf_path, self.pk_chain.get_link_names())

        self.vertices = {}
        removed_links = json.load(open(os.path.join(ROOT_DIR, 'utils/removed_links.json')))[robot_name]
        for link_name, link_mesh in self.meshes.items():
            if link_name in removed_links:  # remove links with small size that do not affect the overall posture
                continue
            v = link_mesh.sample(link_num_points)
            self.vertices[link_name] = v

        self.frame_status = None

    def get_joint_orders(self):
        return [joint.name for joint in self.pk_chain.get_joints()]

    def update_status(self, q):
        if q.shape[-1] != self.dof:
            q = q_rot6d_to_q_euler(q)
        self.frame_status = self.pk_chain.forward_kinematics(q)

    def get_transformed_links_pc(self, q=None, links_pc=None):
        """
        Use robot link pc & q value to get point cloud.

        :param q: (6 + DOF,), joint values (euler representation)
        :param links_pc: {link_name: (N_link, 3)}, robot links pc dict, not None only for get_sampled_pc()
        :return: point cloud: (N, 4), with link index
        """
        if q is None:
            q = torch.zeros(self.dof, dtype=torch.float32, device=self.device)
        self.update_status(q)
        if links_pc is None:
            links_pc = self.links_pc

        all_pc_se3 = []
        for link_index, (link_name, link_pc) in enumerate(links_pc.items()):
            if not torch.is_tensor(link_pc):
                link_pc = torch.tensor(link_pc, dtype=torch.float32, device=q.device)
            n_link = link_pc.shape[0]
            se3 = self.frame_status[link_name].get_matrix()[0].to(q.device)
            homogeneous_tensor = torch.ones(n_link, 1, device=q.device)
            link_pc_homogeneous = torch.cat([link_pc.to(q.device), homogeneous_tensor], dim=1)
            link_pc_se3 = (link_pc_homogeneous @ se3.T)[:, :3]
            index_tensor = torch.full([n_link, 1], float(link_index), device=q.device)
            link_pc_se3_index = torch.cat([link_pc_se3, index_tensor], dim=1)
            all_pc_se3.append(link_pc_se3_index)
        all_pc_se3 = torch.cat(all_pc_se3, dim=0)

        return all_pc_se3

    def get_sampled_pc(self, q=None, num_points=512):
        """
        :param q: (9 + DOF,), joint values (rot6d representation)
        :param num_points: int, number of sampled points
        :return: ((N, 3), list), sampled point cloud (numpy) & index
        """
        if q is None:
            q = self.get_canonical_q()

        sampled_pc = self.get_transformed_links_pc(q, self.vertices)
        return farthest_point_sampling(sampled_pc, num_points)

    def get_canonical_q(self):
        lower, upper = self.pk_chain.get_joint_limits()
        canonical_q = torch.tensor(lower) * 0.75 + torch.tensor(upper) * 0.25
        canonical_q[:6] = 0
        return canonical_q

    def get_initial_q(self, q=None, max_angle: float = math.pi / 6):
        """
        Compute robot initial q value according to target grasp.

        :param q: (6 + DOF,) or (9 + DOF,), joint values (euler/rot6d representation)
        :param max_angle: float, maximum angle of the random rotation
        :return: initial q: (6 + DOF,), rot6d representation
        """
        if q is None:
            q_initial = torch.zeros(self.dof, dtype=torch.float32, device=self.device)

            q_initial[3:6] = (torch.rand(3) * 2 - 1) * torch.pi
            q_initial[5] /= 2

            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            if self.robot_name == 'leaphand':
                q_initial[6:] = (lower_joint_limits + upper_joint_limits) / 2
                q_initial[[-7, -11, -15]] = 0
            else:
                portion = random.uniform(0.65, 0.85)
                q_initial[6:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
        else:
            # target_center = self.get_transformed_links_pc(q).mean(dim=0)[:3]
            # dataset uses rot6d, so for initial_q we use the same shape
            if len(q) == self.dof:
                q = q_euler_to_q_rot6d(q)
            q_initial = q.clone()

            # compute random initial position
            # delta_center = target_center * random.uniform(9, 11)
            # q_initial[:3] += delta_center

            # compute random initial rotation
            # direction = - q_initial[:3] / torch.norm(q_initial[:3])
            # angle = torch.tensor(random.uniform(0, max_angle), device=q.device)  # sample rotation angle
            # axis = torch.randn(3).to(q.device)  # sample rotation axis
            # axis -= torch.dot(axis, direction) * direction  # ensure orthogonality
            # axis = axis / torch.norm(axis)
            # random_rotation = axisangle_to_matrix(axis, angle).to(q.device)
            # rotation_matrix = random_rotation @ rot6d_to_matrix(q_initial[3:9])
            # q_initial[3:9] = matrix_to_rot6d(rotation_matrix)

            # compute random initial joint values
            lower_joint_limits, upper_joint_limits = self.pk_chain.get_joint_limits()
            lower_joint_limits = torch.tensor(lower_joint_limits[6:], dtype=torch.float32)
            upper_joint_limits = torch.tensor(upper_joint_limits[6:], dtype=torch.float32)
            portion = random.uniform(0.65, 0.85)
            q_initial[9:] = lower_joint_limits * portion + upper_joint_limits * (1 - portion)
            # q_initial[9:] = torch.zeros_like(q_initial[9:], dtype=q.dtype, device=q.device)

            q_initial = q_rot6d_to_q_euler(q_initial)

        return q_initial

    def get_trimesh_q(self, q):
        self.update_status(q)

        scene = trimesh.Scene()
        parts = {}
        for link_name in self.vertices:
            mesh_transform_matrix = self.frame_status[link_name].get_matrix()[0].cpu().numpy()
            part_mesh = (self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))
            parts[link_name] = part_mesh
            scene.add_geometry(part_mesh)

        return_dict = {
            'visual': scene,
            'parts': parts
        }
        return return_dict

    def get_trimesh_se3(self, transform, index):
        scene = trimesh.Scene()
        for link_name in transform:
            mesh_transform_matrix = transform[link_name][index].cpu().numpy()
            part_mesh = (self.meshes[link_name].copy().apply_transform(mesh_transform_matrix))
            scene.add_geometry(part_mesh)

        return_dict = {
            'visual': scene
        }
        return return_dict


def create_hand_model(robot_name, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), num_points=512):
    json_path = os.path.join(ROOT_DIR, 'data/data_urdf/robot/urdf_assets_meta.json')
    urdf_assets_meta = json.load(open(json_path))
    urdf_path = os.path.join(ROOT_DIR, urdf_assets_meta['urdf_path'][robot_name])
    meshes_path = os.path.join(ROOT_DIR, urdf_assets_meta['meshes_path'][robot_name])
    links_pc_path = os.path.join(ROOT_DIR, f'data/PointCloud/robot/{robot_name}.pt')
    hand_model = HandModel(robot_name, urdf_path, meshes_path, links_pc_path, device, num_points)
    return hand_model


if __name__ == '__main__':
    hand = create_hand_model('shadowhand')
