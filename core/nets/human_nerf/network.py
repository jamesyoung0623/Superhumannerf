import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import load_mweight_vol_decoder, load_pose_decoder

from datetime import datetime

import tinycudann as tcnn
import numpy as np


class Network(nn.Module):
    def __init__(self, cfg):
        super(Network, self).__init__()
        self.cfg = cfg
        # Skeletal motion ----------------------------------------------
        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(total_bones=self.cfg.total_bones)

        # motion weight volume
        self.mweight_vol_decoder = load_mweight_vol_decoder(self.cfg.mweight_volume.module)(self.cfg)

        # pose correction -------------------------------------------
        # load_pose_decoder:
        self.pose_decoder = load_pose_decoder(cfg.pose_decoder.module)(self.cfg)   
        
        scale=1
        self.scale = scale

        #self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        #self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        #self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        #self.grid_size = 128
        #self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))
        
        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16; 
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        
        # canonical ----------------------------------------------
        # canonical positional encoding
        
        self.cnl_xyz_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=48,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        
        self.cnl_dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        
        self.rgb_net = tcnn.Network(
            n_input_dims=64, n_output_dims=4,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )
        
        
        # Non-rigid motion ----------------------------------------------
        # non-rigid motion st positional encoding
    
        self.non_rigid_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=36,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        
        
        # non-rigid motion MLP
        
        self.non_rigid_net = tcnn.Network(
            n_input_dims=105, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "ReLU",
                "n_neurons": 128,
                "n_hidden_layers": 5,
            }
        )

    def _query_mlp(self, pos_xyz, pos_dir, non_rigid_mlp_input):
        
        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3) 
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])   # dj: [307200, 3]
        dir_flat = torch.reshape(pos_dir, [-1, pos_xyz.shape[-1]])   # dj: [307200, 3]
        chunk = self.cfg.netchunk_per_gpu*len(self.cfg.secondary_gpus)
        
        result = self._apply_mlp_kernels(
            pos_flat=pos_flat,
            dir_flat=dir_flat,
            non_rigid_mlp_input=non_rigid_mlp_input,
            chunk=chunk
        )

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(raws_flat, list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output

    def _expand_input(self, input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))

    def _apply_mlp_kernels(self, pos_flat, dir_flat, non_rigid_mlp_input, chunk):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk): # 307200
            start = i
            end = i + chunk
            end = min(end, pos_flat.shape[0])
            total_elem = end - start
            
            xyz = pos_flat[start:end] # dj: [307200, 3] 3D coordinates
            dir = dir_flat[start:end] # dj: [307200, 3] 3D coordinates' directions
            ### -----------------------------------------
            if not self.cfg.network.ignore_non_rigid_motions:
                non_rigid_embed_xyz = self.non_rigid_encoder(xyz)                               #(307200, 32)
                condition_code = self._expand_input(non_rigid_mlp_input, total_elem)            #(307200, 69)
                non_rigid_input = torch.cat([condition_code, non_rigid_embed_xyz], dim=-1)      #(307200, 101)
                non_rigid_output = self.non_rigid_net(non_rigid_input)                          #(307200, 3)
                xyz = xyz + non_rigid_output

            xyz = (xyz-self.xyz_min)/(self.xyz_max-self.xyz_min)
            pos_embedded = self.cnl_xyz_encoder(xyz)

            dir = dir/torch.norm(dir, dim=1, keepdim=True)
            dir_embedded = self.cnl_dir_encoder((dir+1)/2).cuda()

            rgb_output = self.rgb_net(torch.cat([dir_embedded, pos_embedded], 1))
            cnl_mlp_output = rgb_output

            raws += [cnl_mlp_output]

        return {'raws': torch.cat(raws, dim=0).to(self.cfg.primary_gpus[0])}

    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], self.cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+self.cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        return {k : torch.cat(all_ret[k], 0) for k in all_ret}

    def _raw2outputs(self, raw, raw_mask, z_vals, rays_d, bgcolor=None):
        # raw: [N_rays, N_samples, 4]
        # raw_mask : [N_rays, N_samples, 1]
        # z_vals : [N_rays, N_samples]
        # rays_d : [N_rays, 3]
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[..., :1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)                 # [N_rays, N_samples]

        # [N_rays, N_samples, 3]
        rgb = torch.sigmoid(raw[..., :3])
        # [N_rays, N_samples]
        alpha = 1.0 - torch.exp(-F.relu(raw[..., 3])*dists)
        
        alpha = alpha * raw_mask[:, :, 0]                                      # [N_rays, N_samples]
        weights = alpha * torch.cumprod( torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[..., None] * rgb, -2)                       # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)                            # [N_rays]
        acc_map = torch.sum(weights, -1)                                       # [N_rays]

        rgb_map = rgb_map + (1.-acc_map[..., None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map, alpha

    def _sample_motion_fields(self, pts, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] # [24, 32, 32, 32]

        # motion_scale_Rs [24, 3, 3]; motion_Ts [24, 3]
        # cnl_bbox_min_xyz [3]; cnl_bbox_scale_xyz [3]
        weights_list = []
        pos_list = []
        # dj: mapping from observation space to canonical space in bone-wise 
        
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :] # dj: pos in canonical space
            pos_list.append(pos)
            pos = (pos - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0 
            
            motion_weight = motion_weights[i].unsqueeze(0).unsqueeze(0)
            pos = pos.unsqueeze(0).unsqueeze(0)
            
            weights = F.grid_sample(input=motion_weight, grid=pos, mode='bilinear', padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None] # [1, 1, 1, 1, 307200] to [307200,1]


            weights_list.append(weights) # per canonical pixel's bones weights

        backwarp_motion_weights = torch.cat(weights_list, dim=-1) # dj: [N_rays x N_samples, #bones]
        total_bases = backwarp_motion_weights.shape[-1] # #bones
        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, dim=-1, keepdim=True) # dj: [N_rays x N_samples, 1]

        weighted_motion_fields = []
        for i in range(total_bases):
            pos = pos_list[i]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        x_skel = torch.sum(torch.stack(weighted_motion_fields, dim=0), dim=0) / backwarp_motion_weights_sum.clamp(min=0.0001) # dj: [N_rays x N_samples, 3]
        fg_likelihood_mask = backwarp_motion_weights_sum # dj: [N_rays x N_samples, 1]
        x_skel = x_skel.reshape(orig_shape[:2]+[3]) # dj: [N_rays, N_samples, 3]
        backwarp_motion_weights = backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases]) # dj: [N_rays, N_samples, 24]
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        results = {}
        
        if 'x_skel' in output_list: # [N_rays x N_samples, 3]
            results['x_skel'] = x_skel
        if 'fg_likelihood_mask' in output_list: # [N_rays x N_samples, 1]
            results['fg_likelihood_mask'] = fg_likelihood_mask
        
        return results

    def _unpack_ray_batch(self, ray_batch):
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6] 
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2]) 
        near, far = bounds[..., 0], bounds[..., 1] 
        return rays_o, rays_d, near, far


    def _get_samples_along_ray(self, N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=self.cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, self.cfg.N_samples]) 

    def _stratified_sampling(self, z_vals):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand
        return z_vals


    def _render_rays(self, ray_batch, motion_scale_Rs, motion_Ts, motion_weights_vol, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, non_rigid_mlp_input=None, bgcolor=None, **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far) # dj: [N_rays=2400, nSamples=128]
        
        if self.cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)          # dj: [N_rays=2400, nSamples=128]

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # dj: [2400, 128, 3] (N_rays, N_samples, xyz)

        mv_output = self._sample_motion_fields(
            pts=pts,
            motion_scale_Rs=motion_scale_Rs[0], 
            motion_Ts=motion_Ts[0], 
            motion_weights_vol=motion_weights_vol,
            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz,
            output_list=['x_skel', 'fg_likelihood_mask']
        )

        pts_mask = mv_output['fg_likelihood_mask']
        cnl_pts = mv_output['x_skel']               # dj: [2400, 128, 3] (N_rays, N_samples, xyz)
        
        # cnl_pts: [2400, 128, 3] (N_rays, N_samples, xyz)
        # distortion loss: normalized distances s, normalized weights w 
        # distloss = self._distortion_loss(cnl_pts,)

        pts_dir = rays_d[..., None, :] * torch.ones_like(z_vals[..., :, None])
        # cnl_pts.shape [2400, 128, 3]

        query_result = self._query_mlp(
            pos_xyz=cnl_pts,
            pos_dir=pts_dir,
            non_rigid_mlp_input=non_rigid_mlp_input
        )
        raw = query_result['raws']
        
        rgb_map, acc_map, _, depth_map, alpha = self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)
        
        return {'rgb' : rgb_map, 'alpha' : acc_map, 'depth': depth_map}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts

    def _multiply_corrected_Rs(self, Rs, correct_Rs):
        total_bones = self.cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self, iter_val, rays, dst_Rs, dst_Ts, cnl_gtfms, motion_weights_priors, motionCLIP, dst_posevec=None, near=None, far=None, **kwargs):
        # rays: (2, 2400, 3)
        dst_Rs = dst_Rs[None, ...] # [1, 24, 3, 3]
        dst_Ts = dst_Ts[None, ...] # [1, 24, 3]
        dst_posevec = dst_posevec[None, ...] # [1, 69]
        cnl_gtfms = cnl_gtfms[None, ...]
        motion_weights_priors = motion_weights_priors[None, ...]
        #motionCLIP=motionCLIP[None, ...] # dj [540, 512]

        # correct body pose
        ### -----------------------------------------
        if not self.cfg.network.ignore_pose_correction and iter_val >= self.cfg.pose_decoder.kick_in_iter:
            pose_out = self.pose_decoder(dst_posevec) # [1, 23, 3, 3] axis-angle (3) to rotation matrix (3,3)
            delta_Rs = pose_out['Rs'] # [1, 23, 3, 3]
            delta_Ts = pose_out.get('Ts', None)
           
            dst_Rs_no_root = dst_Rs[:, 1:, ...]
            dst_Rs_no_root = self._multiply_corrected_Rs(dst_Rs_no_root, delta_Rs)
            dst_Rs = torch.cat([dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)
        
            if delta_Ts is not None:
                dst_Ts = dst_Ts + delta_Ts

        # delayed optimization
        if iter_val < self.cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs['non_rigid_mlp_input'] = non_rigid_mlp_input
        
        # skeletal motion and non-rigid motion
        ### -----------------------------------------
        # dj: dst_Rs [1, 24, 3, 3]; dst_Ts [1, 24, 3], cnl_gtfms [1, 24, 4, 4]
        # dj: motion_scale_Rs [1, 24, 3, 3]; motion_Ts [1, 24, 3] MAPPING from Target pose to T-pose
        motion_scale_Rs, motion_Ts = self._get_motion_base(dst_Rs=dst_Rs, dst_Ts=dst_Ts, cnl_gtfms=cnl_gtfms)
        # dj: motion_weights_vol [25, 32, 32, 32] each bone-level (with BG) weights in volume (x,y,z)
        motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors=motion_weights_priors)
        motion_weights_vol = motion_weights_vol[0] # remove batch dimension motion_weights_vol [25, 32, 32, 32]
        
        kwargs['motion_scale_Rs'] = motion_scale_Rs
        kwargs['motion_Ts'] = motion_Ts
        kwargs['motion_weights_vol'] = motion_weights_vol


        ### -----------------------------------------
        rays_o, rays_d = rays
        rays_shape = rays_d.shape # [2400, 3]

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs)

        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret