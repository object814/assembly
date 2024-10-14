import os
import sys
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)
import copy
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.tv_mlp import MLP as TVMLP
from model.vn_dgcnn import VN_DGCNN, VNArgs
from third_party.dcp.model import (
    DGCNN,
    DGCNNClassification,
    Decoder,
    DecoderLayer,
    Encoder,
    EncoderDecoder,
    EncoderLayer,
    MultiHeadedAttention,
    PositionwiseFeedForward
)
from model.pointnet2 import PointNet2, PointNet2Cls


class CustomTransformer(nn.Module):
    """This is a custom transformer model that is used to embed the point clouds.
    It is based on the transformer model from the DCP paper.

    See: https://github.com/WangYueFt/dcp/blob/master/model.py
    """

    def __init__(
        self,
        emb_dims=512,
        n_blocks=1,
        dropout=0.0,
        ff_dims=1024,
        n_heads=4,
        bidirectional=True,
    ):
        super(CustomTransformer, self).__init__()
        self.emb_dims = emb_dims
        self.N = n_blocks
        self.dropout = dropout
        self.ff_dims = ff_dims
        self.n_heads = n_heads
        self.bidirectional = bidirectional
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.emb_dims)
        ff = PositionwiseFeedForward(self.emb_dims, self.ff_dims, self.dropout)
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(self.emb_dims, c(attn), c(ff), self.dropout), self.N),
            Decoder(
                DecoderLayer(self.emb_dims, c(attn), c(attn), c(ff), self.dropout),
                self.N,
            ),
            nn.Sequential(),
            nn.Sequential(),
            nn.Sequential(),
        )

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = self.model(tgt, src, None, None)
        src_attn = self.model.decoder.layers[-1].src_attn.attn

        outputs = {"src_embedding": src_embedding, "src_attn": src_attn}

        if self.bidirectional:
            tgt_embedding = (
                self.model(src, tgt, None, None)
            )
            tgt_attn = self.model.decoder.layers[-1].src_attn.attn

            outputs = {
                **outputs,
                "tgt_embedding": tgt_embedding,
                "tgt_attn": tgt_attn,
            }

        return outputs


class MLPKernel(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.feature_dim = feature_dim
        self.mlp = TVMLP(2 * feature_dim, [300, 100, 1])

    def forward(self, x1, x2):
        v1 = self.mlp(torch.cat([x1, x2], axis=-1))
        v2 = self.mlp(torch.cat([x2, x1], axis=-1))
        return F.softplus((v1 + v2) / 2)

class ResnetBlockFC(nn.Module):
    ''' Fully connected ResNet Block class.
    Args:
        size_in (int): input dimension
        size_out (int): output dimension
        size_h (int): hidden dimension
    '''

    def __init__(self, size_in, size_out=None, size_h=None):
        super().__init__()
        if size_out is None:
            size_out = size_in

        if size_h is None:
            size_h = min(size_in, size_out)

        self.size_in = size_in
        self.size_h = size_h
        self.size_out = size_out

        self.fc_0 = nn.Linear(size_in, size_h)
        self.fc_1 = nn.Linear(size_h, size_out)
        self.actvn = nn.ReLU()

        if size_in == size_out:
            self.shortcut = None
        else:
            self.shortcut = nn.Linear(size_in, size_out, bias=False)
        nn.init.zeros_(self.fc_1.weight)

    def forward(self, x, final_nl=False):
        net = self.fc_0(self.actvn(x))
        dx = self.fc_1(self.actvn(net))
        if self.shortcut is not None:
            x_s = self.shortcut(x)
        else:
            x_s = x
        x_out = x_s + dx
        if final_nl:
            return F.leaky_relu(x_out, negative_slope=0.2)
        return x_out

class LatentEncoder(nn.Module):
    def __init__(self, in_dim, dim, out_dim):
        super().__init__()
        self.block = ResnetBlockFC(size_in=in_dim, size_out=dim, size_h=dim)
        self.fc_mu = nn.Linear(dim, out_dim)
        self.fc_logvar = nn.Linear(dim, out_dim)

    def forward(self, x):
        x = self.block(x, final_nl=True)
        return self.fc_mu(x), self.fc_logvar(x)


def create_embedding_network(cfg, is_robot=False) -> nn.Module:
    if cfg.name == 'dgcnn':
        network = DGCNN(emb_dims=cfg.emb_dims)
        if is_robot and cfg.pretrain is not None:
            print(f"Load embedding network pretrain from '{cfg.pretrain}'.")
            pretrain = cfg.pretrain.split('-')
            network.load_state_dict(
                torch.load(
                    os.path.join(ROOT_DIR, f'output/{pretrain[0]}/state_dict/{pretrain[1]}'),
                    map_location=torch.device('cuda:0'),
                )
            )
    elif cfg.name == 'pointnet':
        pass
        # network = PointNet(out_dim=cfg.emb_dims)
    elif cfg.name == 'pointnet++':
        network = PointNet2(in_dim=3,
                            hidden_dim=256,
                            out_dim=cfg.emb_dims)
    else:
        raise ValueError(f"Unknown embedding network type: {cfg.name}")

    return network


class Network(nn.Module):
    def __init__(
        self,
        cfg,
        mode='train',
    ):
        super(Network, self).__init__()

        self.cfg = cfg
        emb_dims = cfg.encoder.emb_dims# * 2 # global

        self.mode = mode

        self.robot_embedding_network = create_embedding_network(cfg.encoder, is_robot=True)
        self.object_embedding_network = DGCNN(emb_dims=cfg.encoder.emb_dims)
        # self.object_embedding_network = PointNet2(in_dim=3, hidden_dim=256, out_dim=emb_dims)

        self.transformer_robot = CustomTransformer(emb_dims=emb_dims, bidirectional=False)
        self.transformer_object = CustomTransformer(emb_dims=emb_dims, bidirectional=False)

        self.kernel = MLPKernel(emb_dims + 64)

        # self.point_encoder = PointNet2Cls(3 + emb_dims, 128)  # zhenyu config TODO
        self.point_encoder = DGCNNClassification(emb_dims=emb_dims, output_channels=128, feat_dim=emb_dims)
        self.latent_encoder = LatentEncoder(in_dim=128, dim=256, out_dim=64)  # zhenyu config TODO

    def forward(self, robot_pc, object_pc, target_pc=None):
        if self.cfg.center_feature:
            robot_pc = robot_pc - robot_pc.mean(dim=1, keepdim=True)

        robot_embedding = self.robot_embedding_network(robot_pc)
        object_embedding = self.object_embedding_network(object_pc)
        if torch.isnan(robot_embedding).any() or torch.isnan(object_embedding).any():
            print('NaN ckpt 1')
            exit()

        if self.cfg.frozen_embedding:
            robot_embedding = robot_embedding.detach()

        # robot_embedding_global = robot_embedding.mean(dim=1, keepdim=True).repeat(1, robot_pc.shape[1], 1)
        # robot_embedding = torch.cat([robot_embedding, robot_embedding_global], dim=-1)

        transformer_robot_outputs = self.transformer_robot(robot_embedding, object_embedding)
        transformer_object_outputs = self.transformer_object(object_embedding, robot_embedding)
        robot_embedding_tf = robot_embedding + transformer_robot_outputs["src_embedding"]
        object_embedding_tf = object_embedding + transformer_object_outputs["src_embedding"]
        if torch.isnan(robot_embedding_tf).any() or torch.isnan(object_embedding_tf).any():
            print('NaN ckpt 2')
            exit()

        # CVAE
        if self.mode == 'train':
            pc = torch.cat([target_pc, object_pc], dim=1)
            emb = torch.cat([robot_embedding_tf, object_embedding_tf], dim=1)
            latent = self.point_encoder(torch.cat([pc, emb], -1))
            mu, logvar = self.latent_encoder(latent)
            z_dist = torch.distributions.normal.Normal(mu, torch.exp(0.5 * logvar))
            z = z_dist.rsample()  # (B, latent_dim)
        else:
            mu, logvar = None, None
            z = torch.randn(robot_pc.shape[0], 64).to(robot_pc.device)  # zhenyu latent dim config TODO
            # z = torch.zeros(robot_pc.shape[0], 128).to(robot_pc.device)  # zhenyu latent dim config TODO

        z = z.unsqueeze(dim=1).repeat(1, robot_embedding_tf.shape[1], 1)  # (B, N, latent_dim)

        # (B, N, D + latent_dim)
        Phi_A = torch.cat([robot_embedding_tf, z], dim=-1)
        Phi_B = torch.cat([object_embedding_tf, z], dim=-1)

        # compute reldist matrix
        if self.cfg.block_computing:  # use matrix block computation to save GPU memory
            B, N, D = Phi_A.shape
            block_num = 4  # experimental result, reaching a balance between speed and GPU memory
            N_block = N // block_num
            assert N % N_block == 0, 'Unable to perform block computation.'

            reldist = torch.zeros([B, N, N], dtype=torch.float32, device=Phi_A.device)
            for A_i in range(block_num):
                Phi_A_block = Phi_A[:, A_i * N_block: (A_i + 1) * N_block, :]  # (B, N_block, D)
                for B_i in range(block_num):
                    Phi_B_block = Phi_B[:, B_i * N_block: (B_i + 1) * N_block, :]  # (B, N_block, D)

                    Phi_A_r = Phi_A_block.unsqueeze(2).repeat(1, 1, N_block, 1).reshape(B * N_block * N_block, D)
                    Phi_B_r = Phi_B_block.unsqueeze(1).repeat(1, N_block, 1, 1).reshape(B * N_block * N_block, D)

                    reldist[:, A_i * N_block: (A_i + 1) * N_block, B_i * N_block: (B_i + 1) * N_block] \
                        = self.kernel(Phi_A_r, Phi_B_r).reshape(B, N_block, N_block)
        else:
            Phi_A_r = (
                Phi_A.unsqueeze(2)
                .repeat(1, 1, Phi_A.shape[1], 1)
                .reshape(Phi_A.shape[0] * Phi_A.shape[1] * Phi_A.shape[1], Phi_A.shape[2])
            )
            Phi_B_r = (
                Phi_B.unsqueeze(1)
                .repeat(1, Phi_B.shape[1], 1, 1)
                .reshape(Phi_B.shape[0] * Phi_B.shape[1] * Phi_B.shape[1], Phi_B.shape[2])
            )
            reldist = self.kernel(Phi_A_r, Phi_B_r).reshape(Phi_A.shape[0], Phi_A.shape[1], Phi_B.shape[1])

        outputs = {
            'reldist': reldist,
            'mu': mu,
            'logvar': logvar,
        }
        return outputs


def create_network(cfg, mode):
    network = Network(
        cfg=cfg,
        mode=mode
    )
    return network
