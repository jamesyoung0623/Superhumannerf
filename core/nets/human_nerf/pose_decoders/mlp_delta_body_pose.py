import torch
import torch.nn as nn
import tinycudann as tcnn
from core.utils.network_util import initseq, RodriguesModule


class BodyPoseRefiner(nn.Module):
    def __init__(self, cfg):
        super(BodyPoseRefiner, self).__init__()
        self.cfg = cfg
        self.embedding_size = cfg.pose_decoder.embedding_size
        self.mlp_width = cfg.pose_decoder.mlp_width
        self.mlp_depth = cfg.pose_decoder.mlp_depth
        
        # embedding_size: 69; mlp_width: 256; mlp_depth: 4
        block_mlps = [nn.Linear(self.embedding_size, self.mlp_width), nn.ReLU()]
        
        for _ in range(0, self.mlp_depth-1):
            block_mlps += [nn.Linear(self.mlp_width, self.mlp_width), nn.ReLU()]

        self.total_bones = cfg.total_bones - 1
        block_mlps += [nn.Linear(self.mlp_width, 3 * self.total_bones)]

        self.block_mlps = nn.Sequential(*block_mlps)
        initseq(self.block_mlps)

        # init the weights of the last layer as very small value
        # -- at the beginning, we hope the rotation matrix can be identity 
        init_val = 1e-5
        last_layer = self.block_mlps[-1]
        last_layer.weight.data.uniform_(-init_val, init_val)
        last_layer.bias.data.zero_()
        
        self.pose_decoder = tcnn.Network(
            n_input_dims=self.cfg.pose_decoder.embedding_size, n_output_dims=self.cfg.pose_decoder.embedding_size,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": self.cfg.pose_decoder.mlp_width,
                "n_hidden_layers": self.cfg.pose_decoder.mlp_depth
            }
        )

        self.rodriguez = RodriguesModule()
    
        
    def forward(self, pose_input):
        #rvec = self.block_mlps(pose_input).view(-1, 3)
        rvec = self.pose_decoder(pose_input).view(-1, 3)
        Rs = self.rodriguez(rvec).view(-1, self.total_bones, 3, 3)
        return {"Rs": Rs.type(torch.HalfTensor).cuda()}
