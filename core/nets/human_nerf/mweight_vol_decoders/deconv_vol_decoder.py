import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import ConvDecoder3D


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, embedding_size=256, volume_size=32, total_bones=24):
        super(MotionWeightVolumeDecoder, self).__init__()

        self.total_bones = total_bones
        self.volume_size = volume_size
        
        from configs import cfg
        # torch.manual_seed(cfg.train.seed) #######################
        self.const_embedding = nn.Parameter(torch.randn(embedding_size), requires_grad=True )

        self.decoder = ConvDecoder3D(embedding_size=embedding_size, volume_size=volume_size, voxel_channels=total_bones+1)


    def forward(self, motion_weights_priors, **_):
        # motion_weights_priors [1, 25, 32, 32, 32] (body_util.py def approx_gaussian_bone_volumes)
        embedding = self.const_embedding[None, ...] # [1,256]
        decoded_weights = F.softmax(self.decoder(embedding) + torch.log(motion_weights_priors), dim=1) # [1, 25, 32, 32, 32]
        return decoded_weights
# motion_weights_priors.view(1, 25, -1).sum(dim=2).shape