import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import time
import viser
import trimesh
import numpy as np
import torch
from plotly import graph_objects as go
from plotly import io as pio
from scipy.spatial.transform import Rotation as R

from utils.hand_model import create_hand_model

color_list = ['blue', 'orange', 'green', 'purple', 'yellow', 'pink', 'brown', 'darkorange', 'red', 'magenta',
              'gold', 'deepskyblue', 'violet', 'blueviolet', 'darkred', 'cyan', 'darkcyan', 'darkgoldenrod',
              'darkgreen', 'lime', 'maroon', 'navy', 'olive', 'teal', 'aqua', 'silver']

def visualize_pc(pc, object_name=None, title=None):
    """
    :param pc: (N, 3) or (N, 4), robot point cloud to visualize (w/ or w/o link index)
    :param object_name: str, object name to get mesh
    :param title: str, title of the figure
    :return: None
    """
    server = viser.ViserServer(host='127.0.0.1', port=8080)
    #
    # name = object_name.split('+')
    # object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
    # object_trimesh = trimesh.load_mesh(object_path)
    # server.scene.add_mesh_simple(object_name, object_trimesh.vertices, object_trimesh.faces, opacity=0.6)
    # color_map = np.array([
    #     [228, 26, 28],
    #     [55, 126, 184],
    #     [77, 175, 74],
    #     [152, 78, 163],
    #     [255, 127, 0],
    #     [255, 255, 51],
    #     [166, 86, 40],
    #     [247, 129, 191],
    #     [153, 153, 153],
    #     [255, 0, 255],
    #     [0, 255, 255],
    #     [0, 0, 255],
    #     [0, 255, 0],
    #     [255, 0, 0],
    #     [0, 0, 255],
    #     [255, 165, 0],
    #     [255, 192, 203],
    #     [128, 0, 128],
    #     [0, 128, 0],
    #     [0, 0, 128],
    # ])
    # colors = color_map[pc[:, 3].to(torch.int64).numpy()] if pc.shape[1] == 4 else (0, 0, 255)
    server.scene.add_point_cloud('point cloud',
                                 pc[:, :3].numpy(),
                                 point_size=0.001,
                                 point_shape="circle",
                                 colors=(0, 0, 200))
    while True:
        time.sleep(1)

    layout = go.Layout(
        title=title if title is not None else '',
        scene=dict(
            xaxis=dict(title='X'),
            yaxis=dict(title='Y'),
            zaxis=dict(title='Z')
        )
    )
    robot_vis = go.Scatter3d(
        x=pc[:, 0], y=pc[:, 1], z=pc[:, 2],
        mode='markers',
        marker=dict(
            size=5,
            color=[color_list[int(i)] for i in pc[:, 3]] if pc.shape[1] == 4 else 'blue',
            colorscale='Viridis',
            opacity=0.8
        )
    )
    if object_name is not None:
        obj = object_name.split('+')
        mesh_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{obj[0]}/{obj[1]}/{obj[1]}.stl')
        mesh = trimesh.load_mesh(mesh_path)
        x, y, z = zip(*mesh.vertices)
        object_vis = go.Mesh3d(
            x=x,
            y=y,
            z=z,
            i=[face[0] for face in mesh.faces],
            j=[face[1] for face in mesh.faces],
            k=[face[2] for face in mesh.faces],
            color='lightpink',
            opacity=0.6
        )
        fig = go.Figure(data=[robot_vis, object_vis], layout=layout)
    else:
        fig = go.Figure(data=[robot_vis], layout=layout)
    pio.show(fig)

def visualize(hand,
              object_name: str,
              q_dict: dict,
              se3_dict: dict,
              se3_index: int = 0):
    server = viser.ViserServer(host='127.0.0.1', port=8080)

    name = object_name.split('+')
    object_path = os.path.join(ROOT_DIR, f'data/data_urdf/object/{name[0]}/{name[1]}/{name[1]}.stl')
    object_trimesh = trimesh.load_mesh(object_path)
    server.scene.add_mesh_trimesh(object_name, object_trimesh)

    for name, q in q_dict.items():
        robot_trimesh = hand.get_trimesh_q(q)["visual"]
        server.scene.add_mesh_trimesh(name, robot_trimesh, visible=False)

    for name, se3 in se3_dict.items():
        robot_trimesh = hand.get_trimesh_se3(se3, se3_index)["visual"]
        server.scene.add_mesh_trimesh(name, robot_trimesh)

    while True:
        time.sleep(1)

def normalize(x):
    """
    Normalize the input vector. If the magnitude of the vector is zero, a small value is added to prevent division by zero.

    Parameters:
    - x (np.ndarray): Input vector to be normalized.

    Returns:
    - np.ndarray: Normalized vector.
    """
    if len(x.shape) == 1:
        mag = np.linalg.norm(x)
        if mag == 0:
            mag = mag + 1e-10
        return x / mag
    else:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-10, norms)
        return x / norms


def sample_transform_w_normals(
    new_palm_center,
    new_face_vector,
    sample_roll,
    ori_face_vector=np.array([1.0, 0.0, 0.0]),
):
    """
    Compute the transformation matrix from the original palm pose to a new palm pose.

    Parameters:
    - new_palm_center (np.ndarray): The point of the palm center [x, y, z].
    - new_face_vector (np.ndarray): The direction vector representing the new palm facing direction.
    - sample_roll (float): The roll angle in range [0, 2*pi).
    - ori_face_vector (np.ndarray): The original direction vector representing the palm facing direction. Default is [1.0, 0.0, 0.0].

    Returns:
    - rst_transform (np.ndarray): A 4x4 transformation matrix.
    """

    rot_axis = np.cross(ori_face_vector, normalize(new_face_vector))
    rot_axis = rot_axis / (np.linalg.norm(rot_axis) + 1e-16)
    rot_ang = np.arccos(np.clip(np.dot(ori_face_vector, new_face_vector), -1.0, 1.0))

    if rot_ang > 3.1415 or rot_ang < -3.1415:
        rot_axis = (
            np.array([1.0, 0.0, 0.0])
            if not np.isclose(ori_face_vector, np.array([1.0, 0.0, 0.0])).all()
            else np.array([0.0, 1.0, 0.0])
        )

    rot = R.from_rotvec(rot_ang * rot_axis).as_matrix()
    roll_rot = R.from_rotvec(sample_roll * new_face_vector).as_matrix()

    final_rot = roll_rot @ rot
    rst_transform = np.eye(4)
    rst_transform[:3, :3] = final_rot
    rst_transform[:3, 3] = new_palm_center
    return rst_transform

def vis_vector(
        start_point,
        vector,
        length=0.1,
        cyliner_r=0.003,
        color=[255, 255, 100, 245],
        no_arrow=False,
):
    """
    start_points: np.ndarray, shape=(3,)
    vectors: np.ndarray, shape=(3,)
    length: cylinder length
    """
    normalized_vector = normalize(vector)
    end_point = start_point + length * normalized_vector

    # create a mesh for the force
    force_cylinder = trimesh.creation.cylinder(
        radius=cyliner_r, segment=np.array([start_point, end_point])
    )

    # create a mesh for the arrowhead
    cone_transform = sample_transform_w_normals(
        end_point, normalized_vector, 0, ori_face_vector=np.array([0.0, 0.0, 1.0])
    )
    arrowhead_cone = trimesh.creation.cone(
        radius=2 * cyliner_r, height=4 * cyliner_r, transform=cone_transform
    )
    # combine the two meshes into one
    if not no_arrow:
        force_mesh = force_cylinder + arrowhead_cone
    else:
        force_mesh = force_cylinder
    force_mesh.visual.face_colors = color

    return force_mesh


if __name__ == '__main__':
    robot_name = 'shadowhand'
    object_name = 'ycb+hammer'
    data_len = 10

    hand = create_hand_model(robot_name, device='cpu')
    dataset_path = os.path.join(ROOT_DIR, f'data/MultiDex_filtered/{robot_name}/{robot_name}.pt')
    metadata = torch.load(dataset_path)['metadata']
    q_list = [m[0] for m in metadata if m[1] == object_name][:data_len]
    q_dict = {}
    for i, q in enumerate(q_list):
        q_dict[f'{robot_name}_{i}'] = q
    visualize(hand, object_name=object_name, q_dict=q_dict, se3_dict={})