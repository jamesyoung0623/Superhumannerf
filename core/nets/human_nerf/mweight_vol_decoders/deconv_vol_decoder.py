import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import ConvDecoder3D


class MotionWeightVolumeDecoder(nn.Module):
    def __init__(self, cfg):
        super(MotionWeightVolumeDecoder, self).__init__()
<<<<<<< HEAD

        self.total_bones = total_bones
        self.volume_size = volume_size
        self.const_embedding = nn.Parameter(torch.randn(embedding_size), requires_grad=True)
        self.decoder = ConvDecoder3D(embedding_size=embedding_size, volume_size=volume_size, voxel_channels=total_bones+1)
=======
        self.embedding_size = cfg.mweight_volume.embedding_size
        self.volume_size = cfg.mweight_volume.volume_size
        self.total_bones = cfg.total_bones
        self.const_embedding = nn.Parameter(torch.randn(self.embedding_size), requires_grad=True)
        self.decoder = ConvDecoder3D(embedding_size=self.embedding_size, volume_size=self.volume_size, voxel_channels=self.total_bones+1)
>>>>>>> 7c4f382b0ad576cfb3ca27df9fda26aad4f0386c


    def forward(self, motion_weights_priors, **_):
        # motion_weights_priors [1, 25, 32, 32, 32] (body_util.py def approx_gaussian_bone_volumes)
        embedding = self.const_embedding[None, ...] # [1,256]
        decoded_weights = F.softmax(self.decoder(embedding) + torch.log(motion_weights_priors), dim=1) # [1, 25, 32, 32, 32]
        return decoded_weights