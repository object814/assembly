import os
import sys
import numpy as np
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT_DIR)

import warnings
import hydra
import torch

from model.network import create_network
from utils.multilateration import multilateration
from utils.se3_transform import compute_se3_transform
from utils.optimization import *
from utils.hand_model import create_hand_model

from controller import controller
import viser 
from diff_robot_hand.hand_model import LeapHandRight
from scipy.spatial.transform import Rotation as R

@hydra.main(version_base="1.2", config_path="", config_name="validate")
def main(cfg):
    device = torch.device(f'cuda:{cfg.gpu}')
    batch_size = cfg.dataset.batch_size
    
    print(f"Device: {device}")
    print('Name:', cfg.name)
    vis_hand = LeapHandRight(
        load_balls_urdf=False,
        load_visual_mesh=True,
        load_col_mesh=True,
    )

    network = create_network(cfg.model, mode='validate').to(device)
    network.load_state_dict(torch.load(f"epoch_28.pth", map_location=device))
    network.eval()

    robot_name = "leaphand"
    ball = trimesh.creation.icosphere(subdivisions=3, radius=0.05)
    object_pc = ball.sample(512)
    object_pc = torch.tensor(object_pc, dtype=torch.float32, device=device).unsqueeze(0)

    hand = create_hand_model(robot_name, device=device)
    initial_q = hand.get_canonical_q().to(device)
    
    # initial_q[3:6] for rotation 
    initial_q[4] = np.pi/2
    robot_pc = hand.get_transformed_links_pc(initial_q)[:, :3].unsqueeze(0)
    
    with torch.no_grad():
        reldist = network(
            robot_pc,  # (B, N, 3)
            object_pc, # (B, M, 3)
        )['reldist'].detach()

        mlat_pc = multilateration(reldist, object_pc)  # (B, N, 3)
        transform, _ = compute_link_pose(hand.links_pc, mlat_pc, is_train=False)  #
        optim_transform = process_transform(hand.pk_chain, transform)

        layer = create_problem(hand.pk_chain, optim_transform.keys())
        predict_q = optimization(hand.pk_chain, layer, initial_q.unsqueeze(0), optim_transform)
        outer_q, inner_q = controller(robot_name, predict_q)  
        print(outer_q, inner_q)
        
    hand_mesh = vis_hand.get_hand_trimesh(predict_q)["visual"]
    hand_mesh_inner = vis_hand.get_hand_trimesh(inner_q)["visual"]
    hand_mesh_outer = vis_hand.get_hand_trimesh(outer_q)["visual"]
    
    sv = viser.ViserServer()
    sv.scene.add_mesh_trimesh("hand", hand_mesh)
    sv.scene.add_mesh_trimesh("hand_inner", hand_mesh_inner)
    sv.scene.add_mesh_trimesh("hand_outer", hand_mesh_outer)
    sv.scene.add_mesh_trimesh("object", ball)
    
    breakpoint()


if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)
    torch.set_num_threads(8)
    main()
