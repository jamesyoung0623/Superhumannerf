import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import load_mweight_vol_decoder, load_pose_decoder

import tinycudann as tcnn
import numpy as np

from torch_efficient_distloss import eff_distloss

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
        # self.pose_decoder = load_pose_decoder(cfg.pose_decoder.module)(self.cfg)   
        
        self.scale = 1

        #self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*self.scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*self.scale)
        #self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        #self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        #self.grid_size = 128
        #self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))
        
        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16; 
        b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        
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
                "n_hidden_layers": 1
            }
        )
        
        self.cnl_dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4
            }
        )
        
        self.rgb_net = tcnn.Network(
            n_input_dims=64, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
        
        self.sigma_net = tcnn.Network(
            n_input_dims=48, n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2
            }
        )
        
        
        # Non-rigid motion ----------------------------------------------
        # non-rigid motion st positional encoding
        
        # self.non_rigid_encoder = tcnn.NetworkWithInputEncoding(
        #     n_input_dims=3, n_output_dims=36,
        #     encoding_config={
        #         "otype": "Grid",
        #         "type": "Hash",
        #         "n_levels": L,
        #         "n_features_per_level": F,
        #         "log2_hashmap_size": log2_T,
        #         "base_resolution": N_min,
        #         "per_level_scale": b,
        #         "interpolation": "Linear"
        #     },
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "ReLU",
        #         "n_neurons": 64,
        #         "n_hidden_layers": 1
        #     }
        # )
        
        # # non-rigid motion MLP
        # self.non_rigid_net = tcnn.Network(
        #     n_input_dims=105, n_output_dims=3,
        #     network_config={
        #         "otype": "FullyFusedMLP",
        #         "activation": "ReLU",
        #         "output_activation": "ReLU",
        #         "n_neurons": self.cfg.non_rigid_motion_mlp.mlp_width,
        #         "n_hidden_layers": self.cfg.non_rigid_motion_mlp.mlp_depth
        #     }
        # )
        

    def _query_mlp(self, pos_xyz, pos_dir, non_rigid_mlp_input):
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])   # dj: [307200, 3]
        dir_flat = torch.reshape(pos_dir, [-1, pos_xyz.shape[-1]])   # dj: [307200, 3]
        chunk = self.cfg.netchunk_per_gpu
        
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
        for i in range(0, pos_flat.shape[0], chunk):
            start = i
            end = i + chunk
            end = min(end, pos_flat.shape[0])
            total_elem = end - start
            
            xyz = pos_flat[start:end]
            dir = dir_flat[start:end]
            ### -----------------------------------------
            if not self.cfg.network.ignore_non_rigid_motions:
                non_rigid_embed_xyz = self.non_rigid_encoder(xyz)                               #(307200, 36)
                condition_code = self._expand_input(non_rigid_mlp_input, total_elem)            #(307200, 69)
                non_rigid_input = torch.cat([condition_code, non_rigid_embed_xyz], dim=-1)      #(307200, 105)
                non_rigid_output = self.non_rigid_net(non_rigid_input)                          #(307200, 3)
                xyz = xyz + non_rigid_output

            xyz = (xyz-self.xyz_min)/(self.xyz_max-self.xyz_min)
            pos_embedded = self.cnl_xyz_encoder(xyz)

            dir = dir/torch.norm(dir, dim=1, keepdim=True)
            dir_embedded = self.cnl_dir_encoder((dir+1)/2).cuda()

            rgb_output = self.rgb_net(torch.cat([dir_embedded, pos_embedded], dim=1))
            sigma_output = self.sigma_net(pos_embedded)

            cnl_mlp_output = torch.cat([rgb_output, sigma_output], dim=1)

            raws += [cnl_mlp_output]

        return {'raws': torch.cat(raws, dim=0).cuda()}
    

    def _sample_motion_fields(self, pts, motion_scale_Rs, motion_Ts, cnl_bbox_min_xyz, cnl_bbox_scale_xyz):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3)

        motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors=self.motion_weights_priors)
        motion_weights_vol = motion_weights_vol[0]
        motion_weights = motion_weights_vol[:-1]

        weights_list = []
        pos_list = []
        
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :] # dj: pos in canonical space
            pos_list.append(pos)
            pos = (pos - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0 
            
            motion_weight = motion_weights[i].unsqueeze(0).unsqueeze(0)
            
            while len(pos.shape) != 5:
                pos = pos.unsqueeze(0)
            
            weights = F.grid_sample(input=motion_weight, grid=pos, mode='bilinear', padding_mode='zeros', align_corners=True)
            weights = weights[0, 0, 0, 0, :, None]

            weights_list.append(weights) # per canonical pixel's bones weights

        backwarp_motion_weights = torch.cat(weights_list, dim=-1)
        total_bases = backwarp_motion_weights.shape[-1]
        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, dim=-1, keepdim=True)

        weighted_motion_fields = []
        for i in range(total_bases):
            pos = pos_list[i]
            weighted_pos = backwarp_motion_weights[:, i:i+1] * pos
            weighted_motion_fields.append(weighted_pos)
        
        x_skel = torch.sum(torch.stack(weighted_motion_fields, dim=0), dim=0) / backwarp_motion_weights_sum.clamp(min=0.0001)
        fg_likelihood_mask = backwarp_motion_weights_sum
        x_skel = x_skel.reshape(orig_shape[:2]+[3])
        backwarp_motion_weights = backwarp_motion_weights.reshape(orig_shape[:2]+[total_bases])
        fg_likelihood_mask = fg_likelihood_mask.reshape(orig_shape[:2]+[1])

        return x_skel, fg_likelihood_mask

    def _unpack_ray_batch(self, ray_batch):
        rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6] 
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2]) 
        near, far = bounds[..., 0], bounds[..., 1] 
        return rays_o, rays_d, near, far


    def _get_samples_along_ray(self, near, far):
        t_vals = torch.linspace(0., 1., steps=self.cfg.N_samples).cuda()
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals

    def _stratified_sampling(self, z_vals):
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape).cuda()
        z_vals = lower + (upper - lower) * t_rand
        return z_vals

    def _raw2outputs(self, raw, raw_mask, z_vals, rays_d, bgcolor):
        def rgb_activation(rgb):
            return torch.sigmoid(rgb)
        
        def sigma_activation(sigma):
            #alpha_init = 1e-6
            #b = np.log(1/np.power((1-alpha_init), 6.25) - 1)            # low-density initialization 
            #b = 1e-6            # low-density initialization 
            #return torch.log(1 + torch.exp(sigma+b))
            return F.relu(sigma)

        dists = z_vals[..., 1:] - z_vals[..., :-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[..., :1].shape).cuda()
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[..., None, :], dim=-1)                 

        rgb = rgb_activation(raw[..., :3])
        sigma = sigma_activation(raw[..., 3])

        alpha = 1.0 - torch.exp(-sigma*dists)
        alpha = alpha * raw_mask[:, :, 0]    

        weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.0-alpha+1e-10], dim=1), dim=1)[:, :-1]
       
        rgb_map = torch.sum(weights[..., None] * rgb, dim=1)                      
        depth_map = torch.sum(weights * z_vals, dim=1)                           
        weights_sum = torch.sum(weights, dim=1)                                      

        rgb_map = rgb_map + (1.0-weights_sum[..., None]) * bgcolor[None, :]/255.

        return rgb_map, depth_map, weights, weights_sum, alpha, sigma

    def _render_rays(self, rays_o, rays_d, near, far, motion_scale_Rs, motion_Ts, cnl_bbox_min_xyz, cnl_bbox_scale_xyz, non_rigid_mlp_input=None, bgcolor=None, **kwargs):
        z_vals = self._get_samples_along_ray(near, far)
        
        if self.cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)      

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

        cnl_pts, pts_mask = self._sample_motion_fields(
            pts=pts,
            motion_scale_Rs=motion_scale_Rs[0], 
            motion_Ts=motion_Ts[0], 
            cnl_bbox_min_xyz=cnl_bbox_min_xyz, 
            cnl_bbox_scale_xyz=cnl_bbox_scale_xyz
        )

        pts_dir = rays_d[..., None, :] * torch.ones_like(z_vals[..., :, None])

        query_result = self._query_mlp(
            pos_xyz=cnl_pts,
            pos_dir=pts_dir,
            non_rigid_mlp_input=non_rigid_mlp_input
        )
        raw = query_result['raws']
        
        rgb_map, depth_map, weights, weights_sum, alpha, sigma = self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)
        
        B = self.cfg.patch.N_patches*(self.cfg.patch.size**2)
        N = 128
  
        m = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        m = torch.cat([z_vals[..., :1], m], -1)

        interval = 1/N
        distloss = eff_distloss(weights, m, interval)

        return {'rgb': rgb_map, 'depth': depth_map, 'weights': weights, 'weights_sum': weights_sum, 'alpha': alpha, 'sigma': sigma}, distloss


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(dst_Rs, dst_Ts, cnl_gtfms)
        return motion_scale_Rs, motion_Ts

    def _multiply_corrected_Rs(self, Rs, correct_Rs):
        Rs = Rs.type(torch.HalfTensor).cuda()
        correct_Rs = correct_Rs.type(torch.HalfTensor).cuda()
        total_bones = self.cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self, iter_val, rays, dst_Rs, dst_Ts, cnl_gtfms, motion_weights_priors, dst_posevec=None, near=None, far=None, **kwargs):
        dst_Rs = dst_Rs[None, ...] # [1, 24, 3, 3]
        dst_Ts = dst_Ts[None, ...] # [1, 24, 3]
        dst_posevec = dst_posevec[None, ...] # [1, 69]

        cnl_gtfms = cnl_gtfms[None, ...]
        self.motion_weights_priors = motion_weights_priors[None, ...]
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
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs['non_rigid_mlp_input'] = non_rigid_mlp_input
        
        # skeletal motion and non-rigid motion
        ### -----------------------------------------
        # dj: dst_Rs [1, 24, 3, 3]; dst_Ts [1, 24, 3], cnl_gtfms [1, 24, 4, 4]
        # dj: motion_scale_Rs [1, 24, 3, 3]; motion_Ts [1, 24, 3] MAPPING from Target pose to T-pose
        motion_scale_Rs, motion_Ts = self._get_motion_base(dst_Rs=dst_Rs, dst_Ts=dst_Ts, cnl_gtfms=cnl_gtfms)
        
        kwargs['motion_scale_Rs'] = motion_scale_Rs
        kwargs['motion_Ts'] = motion_Ts

        ### -----------------------------------------
        rays_o, rays_d = rays
        rays_shape = rays_d.shape
        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()

        all_ret = {}
        total_distloss = None
        for i in range(0, rays_o.shape[0], self.cfg.chunk):
            ret, distloss = self._render_rays(rays_o[i:i+self.cfg.chunk], rays_d[i:i+self.cfg.chunk], near[i:i+self.cfg.chunk], far[i:i+self.cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

            if total_distloss == None:
                total_distloss = distloss
            else:
                total_distloss += distloss

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}

        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret, total_distloss 
