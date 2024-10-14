import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import torch


def calculate_depth(robot_pc, object_names):
    """
    Calculate the average penetration depth of predicted pc into the object.

    :param robot_pc: (B, N, 3)
    :param object_name: list<str>, len = B
    :return: calculated depth, (B,)
    """
    object_pc_list = []
    normals_list = []
    for object_name in object_names:
        name = object_name.split('+')
        object_path = os.path.join(ROOT_DIR, f'data/PointCloud/object/{name[0]}/{name[1]}.pt')
        object_pc_normals = torch.load(object_path).to(robot_pc.device)
        object_pc_list.append(object_pc_normals[:, :3])
        normals_list.append(object_pc_normals[:, 3:])
    object_pc = torch.stack(object_pc_list, dim=0)
    normals = torch.stack(normals_list, dim=0)

    distance = torch.cdist(robot_pc, object_pc)
    distance, index = torch.min(distance, dim=-1)
    index = index.unsqueeze(-1).repeat(1, 1, 3)
    object_pc_indexed = torch.gather(object_pc, dim=1, index=index)
    normals_indexed = torch.gather(normals, dim=1, index=index)
    get_sign = torch.vmap(torch.vmap(lambda x, y: torch.where(torch.dot(x, y) >= 0, 1, -1)))
    signed_distance = distance * get_sign(robot_pc - object_pc_indexed, normals_indexed)
    signed_distance[signed_distance > 0] = 0
    return -torch.mean(signed_distance)