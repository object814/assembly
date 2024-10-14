import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import time
import math
import random
import argparse
import numpy as np
import torch
import viser
import trimesh
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from utils.hand_model import create_hand_model
from utils.se3_transform import compute_link_pose
from utils.rotation import *
from utils.visualization import *


def process_transform(pk_chain, transform, device=None):
    """Compute extra link transform, convert SE3 transform to only translation."""
    new_transform = transform.copy()
    for name in pk_chain.get_frame_names(exclude_fixed=False):
        if name.startswith('extra'):
            frame = pk_chain.find_frame(name)
            parent_name = pk_chain.idx_to_frame[pk_chain.parents_indices[pk_chain.frame_to_idx[name]][-2].item()]
            new_transform[name] = new_transform[parent_name] @ frame.joint.offset.get_matrix()[0]
    for name, se3 in new_transform.items():
        new_transform[name] = se3[:, :3, 3]
        if device is not None:
            new_transform[name] = new_transform[name].to(device)

    return new_transform

def jacobian(pk_chain, q, frame_X_dict, frame_names):
    """
    Calculate Jacobian (dX/dq) of all frames

    Notation: (similar as https://manipulation.csail.mit.edu/pick.html#monogram)
        J: jacobian, X: transform, R: rotation, p: position, v: velocity, w: angular velocity
        <>_BA_C: <X, R, p, w> of frame A measured from frame B expressed in frame C
        W: world frame, J: joint frame, F: link frame

    :param pk_chain: get from pk.build_chain_from_urdf()
    :param q: (6 + DOF,) or (B, 6 + DOF), joint values (euler representation)
    :return: Jacobian: {frame_name: (B, 6, num_joints)}
    """
    jacobian_dict = {}

    q = torch.atleast_2d(q)
    batch_size = q.shape[0]
    joint_names = pk_chain.get_joint_parameter_names()
    num_joints = len(joint_names)
    joint_name2idx = {name: idx for idx, name in enumerate(joint_names)}

    frames = [pk_chain.find_frame(name) for name in pk_chain.get_joint_parent_frame_names()]
    idx = lambda frame: joint_name2idx[frame.joint.name]

    transfer_X = {}
    for frame in frames:
        q_frame = q[:, idx(frame)]
        if frame.joint.joint_type == 'prismatic':
            q_frame = q_frame.unsqueeze(-1)
        transfer_X[idx(frame)] = frame.get_transform(q_frame).get_matrix()
    # transfer_X = {idx(frame): frame.get_transform(q[:, idx(frame)]).get_matrix() for frame in frames}

    frame_X_dict = {f: frame_X_dict[f] for f in frame_X_dict if f in frame_names}

    for frame_name, frame_X in frame_X_dict.items():
        jacobian = torch.zeros((batch_size, 6, num_joints), dtype=pk_chain.dtype, device=pk_chain.device)

        R_WF = frame_X.get_matrix()[:, :3, :3]
        X_JF = torch.eye(4, dtype=pk_chain.dtype, device=pk_chain.device).repeat(batch_size, 1, 1)
        for frame_idx in reversed(pk_chain.parents_indices[pk_chain.frame_to_idx[frame_name]].tolist()):
            frame = pk_chain.find_frame(pk_chain.idx_to_frame[frame_idx])
            joint = frame.joint
            if joint.joint_type == 'fixed':
                if joint.offset is not None:
                    X_JF = joint.offset.get_matrix() @ X_JF
                continue

            R_FJ = X_JF[:, :3, :3].mT
            R_WJ = R_WF @ R_FJ
            p_JF_J = X_JF[:, :3, 3][:, :, None]
            w_WJ_J = joint.axis[None, :, None].repeat(batch_size, 1, 1)
            if joint.joint_type == 'revolute':
                jacobian_v = R_WJ @ torch.cross(w_WJ_J, p_JF_J, dim=1)
                jacobian_w = R_WJ @ w_WJ_J
            elif joint.joint_type == 'prismatic':
                jacobian_v = R_WJ @ w_WJ_J
                jacobian_w = torch.zeros([batch_size, 3, 1], dtype=jacobian_v.dtype, device=jacobian_v.device)
            else:
                raise NotImplementedError(f"Unknown joint_type: {joint.joint_type}")

            joint_idx = joint_name2idx[joint.name]
            X_JF = transfer_X[joint_idx] @ X_JF
            jacobian[:, :, joint_idx] = torch.cat([jacobian_v[..., 0], jacobian_w[..., 0]], dim=1)

        jacobian_dict[frame_name] = jacobian
    return jacobian_dict

def create_problem(pk_chain, frame_names):
    """
    Only use all frame positions (ignore rotation) to optimize joint values.

    :param pk_chain: get from pk.build_chain_from_urdf()
    :param frame_names: list of frame names to optimize
    :return: CvxpyLayer()
    """
    n_joint = len(pk_chain.get_joint_parameter_names())

    delta_q = cp.Variable(n_joint)

    q = cp.Parameter(n_joint)
    jacobian = {}
    frame_xyz = {}
    target_frame_xyz = {}

    objective_expr = 0
    for link_name in frame_names:
        frame_xyz[link_name] = cp.Parameter(3)
        target_frame_xyz[link_name] = cp.Parameter(3)

        jacobian[link_name] = cp.Parameter((3, n_joint))
        delta_frame_xyz = jacobian[link_name] @ delta_q

        predict_frame_xyz = frame_xyz[link_name] + delta_frame_xyz
        objective_expr += cp.norm2(predict_frame_xyz - target_frame_xyz[link_name])
    objective = cp.Minimize(objective_expr)

    lower_joint_limits, upper_joint_limits = pk_chain.get_joint_limits()
    upper_limit = cp.minimum(0.5, upper_joint_limits - q)
    lower_limit = cp.maximum(-0.5, lower_joint_limits - q)
    constraints = [delta_q <= upper_limit, delta_q >= lower_limit]
    problem = cp.Problem(objective, constraints)

    layer = CvxpyLayer(
        problem,
        parameters=[q,
                    *frame_xyz.values(),
                    *target_frame_xyz.values(),
                    *jacobian.values()],
        variables=[delta_q]
    )
    return layer

def optimization(pk_chain, layer, initial_q, transform, n_iter=64):
    if initial_q.shape[-1] != len(pk_chain.get_joint_parameter_names()):
        initial_q = q_rot6d_to_q_euler(initial_q)
    q = initial_q.clone()

    for i in range(n_iter):
        status = pk_chain.forward_kinematics(q)
        jacobians = jacobian(pk_chain, q, status, transform.keys())

        frame_xyz = {}
        target_frame_xyz = {}
        jacobians_xyz = {}
        for link_name, link_jacobian in jacobians.items():
            frame_xyz[link_name] = status[link_name].get_matrix()[:, :3, 3]
            target_frame_xyz[link_name] = transform[link_name]
            jacobians_xyz[link_name] = link_jacobian[:, :3, :]

        delta_q = layer(
            q,
            *list(frame_xyz.values()),
            *list(target_frame_xyz.values()),
            *list(jacobians_xyz.values()),
        )
        q += delta_q[0]
        # print(f'Step {i}:, delta_q norm: {delta_q[0].norm()}')
        if delta_q[0].norm() < 0.3:
            # print("Converged at iteration:", i)
            break
    return q


if __name__ == '__main__':
    robot_name = 'leaphand'
    hand = create_hand_model(robot_name)

    if robot_name == 'leaphand':
        dataset_path = os.path.join(ROOT_DIR, f'data/RealWorldData/dataset_filtered.pt')  # leaphand
    else:
        dataset_path = os.path.join(ROOT_DIR, f'data/CMapDataset_filtered/cmap_dataset_filtered.pt')
    metadata = torch.load(dataset_path)['metadata']
    q_list = [m[0] for m in metadata if m[2] == robot_name]
    random.shuffle(q_list)
    count = 0
    q_initial_list = []
    q_target_list = []
    q_predict_list = []
    for idx, target_q in enumerate(q_list[:50]):
        q_target_list.append(target_q)

        robot_pc = hand.get_transformed_links_pc(target_q)[:, :3].unsqueeze(0)

        transform, _ = compute_link_pose(hand.links_pc, robot_pc, is_train=False)
        optim_transform = process_transform(hand.pk_chain, transform)

        # optim_transform_new = {}
        # for k, v in optim_transform.items():
        #     if not k.startswith('thumb'):
        #         optim_transform_new[k] = v

        layer = create_problem(hand.pk_chain, optim_transform.keys())

        lower, upper = hand.pk_chain.get_joint_limits()
        initial_q = (torch.tensor(lower, dtype=torch.float32) + torch.tensor(upper, dtype=torch.float32)) / 2
        initial_q[[-7, -11, -15]] = 0.
        initial_pc = hand.get_transformed_links_pc(initial_q)[:, :3].unsqueeze(0)

        from utils.se3_transform import compute_se3_transform
        transform = compute_se3_transform(initial_pc, robot_pc)[0]
        initial_q[:3] = transform[:3, 3]
        initial_q[3:6] = matrix_to_euler(transform[:3, :3])
        q_initial_list.append(initial_q)

        t0 = time.time()
        predict_q = optimization(hand.pk_chain, layer, initial_q.unsqueeze(0), optim_transform)
        q_predict_list.append(predict_q)
        t1 = time.time()
        print(f'Robot: {robot_name}, Time: {t1 - t0:.2f} s')
        predict_pc = hand.get_transformed_links_pc(predict_q)[:, :3]

        torch.set_printoptions(precision=3)
        # print(target_q)
        # print(predict_q)
        sum_error = (target_q - predict_q).sum()
        print(idx, sum_error)
        # if abs(sum_error) > 0.5 and abs(sum_error - 6.28) > 0.5 and abs(sum_error + 6.28) > 0.5:
        #     count += 1
        #     print('*************************ERROR!!!!!!!!****************************')
        #     break
    # print(count)
    # exit()

    # new_transform = transform.copy()
    # for name in hand.pk_chain.get_frame_names(exclude_fixed=False):
    #     if name.startswith('extra'):
    #         frame = hand.pk_chain.find_frame(name)
    #         parent_name = hand.pk_chain.idx_to_frame[hand.pk_chain.parents_indices[hand.pk_chain.frame_to_idx[name]][-2].item()]
    #         new_transform[name] = new_transform[parent_name] @ frame.joint.offset.get_matrix()[0]

    server = viser.ViserServer(host='127.0.0.1', port=8080)

    # for link_name, link_se3 in new_transform.items():
    #     wxyz = trimesh.transformations.quaternion_from_matrix(link_se3[0])
    #     xyz = link_se3[0, :3, 3]
    #     server.scene.add_frame(
    #         link_name,
    #         wxyz=wxyz,
    #         position=xyz,
    #         axes_length=0.05,
    #         axes_radius=0.0025,
    #         visible=False
    #     )

    # robot_trimesh = hand.get_trimesh_q(initial_q)['visual']
    # server.scene.add_mesh_trimesh('robot_initial', robot_trimesh, visible=True)

    # robot_trimesh = hand.get_trimesh_q(target_q)['visual']
    # server.scene.add_mesh_trimesh('robot_target', robot_trimesh, visible=True)
    #
    # robot_trimesh = hand.get_trimesh_q(predict_q)['visual']
    # server.scene.add_mesh_trimesh('robot_predict', robot_trimesh, visible=True)


    def update(step):
        robot_trimesh = hand.get_trimesh_q(q_initial_list[step])['visual']
        server.scene.add_mesh_trimesh('robot_initial', robot_trimesh, visible=True)

        robot_trimesh = hand.get_trimesh_q(q_target_list[step])['visual']
        server.scene.add_mesh_trimesh('robot_target', robot_trimesh, visible=True)

        robot_trimesh = hand.get_trimesh_q(q_predict_list[step])['visual']
        server.scene.add_mesh_trimesh('robot_predict', robot_trimesh, visible=True)


    slider = server.gui.add_slider(
        label='step',
        min=0,
        max=49,
        step=1,
        initial_value=0
    )
    slider.on_update(lambda _: update(slider.value))

    # for step, q_step in enumerate(q_list):
    #     robot_trimesh = hand.get_trimesh_q(q_step)['visual']
    #     server.scene.add_mesh_trimesh(f'step_{step}', robot_trimesh, visible=False)

    # server.scene.add_point_cloud(
    #     'initial',
    #     robot_pc[0].numpy(),
    #     point_size=0.001,
    #     point_shape="circle",
    #     colors=(0, 0, 200)
    # )
    # server.scene.add_point_cloud(
    #     'predict',
    #     predict_pc.numpy(),
    #     point_size=0.001,
    #     point_shape="circle",
    #     colors=(200, 0, 0)
    # )

    while True:
        time.sleep(1)
