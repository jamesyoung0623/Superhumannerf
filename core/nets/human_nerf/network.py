import torch
import torch.nn as nn
import torch.nn.functional as F

from core.utils.network_util import MotionBasisComputer
from core.nets.human_nerf.component_factory import load_positional_embedder, load_canonical_mlp, load_mweight_vol_decoder, load_pose_decoder, load_non_rigid_motion_mlp
from core.nets.human_nerf.encoding import get_encoder  # dj

from configs import cfg
from datetime import datetime

import tinycudann as tcnn
import numpy as np
from ngp_pl.models.custom_functions import TruncExp


#from torch import Tensor
#import nerfacc 
#from nerfacc.estimators.occ_grid import OccGridEstimator
#from radiance_fields.ngp import NGPRadianceField

#from utils import (
#    MIPNERF360_UNBOUNDED_SCENES,
#    NERF_SYNTHETIC_SCENES,
#    render_image_with_occgrid,
#    render_image_with_occgrid_test,
#    set_random_seed,
#)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # Skeletal motion ----------------------------------------------
        # motion basis computer
        self.motion_basis_computer = MotionBasisComputer(total_bones=cfg.total_bones)

        # motion weight volume
        # load_mweight_vol_decoder: (core/nets/human_nerf/mweight_vol_decoders/deconv_vol_decoder.py)
        self.mweight_vol_decoder = load_mweight_vol_decoder(cfg.mweight_volume.module)(
            embedding_size=cfg.mweight_volume.embedding_size,
            volume_size=cfg.mweight_volume.volume_size,
            total_bones=cfg.total_bones
        )

        # Non-rigid motion ----------------------------------------------
        # non-rigid motion st positional encoding
        # load_positional_embedder: (core/nets/human_nerf/embedders/hannw_fourier.py)
        self.get_non_rigid_embedder = load_positional_embedder(cfg.non_rigid_embedder.module)

        # non-rigid motion MLP
        _, non_rigid_pos_embed_size = self.get_non_rigid_embedder(cfg.non_rigid_motion_mlp.multires, cfg.non_rigid_motion_mlp.i_embed)

        # load_non_rigid_motion_mlp: (core/nets/human_nerf/non_rigid_motion_mlps/mlp_offset.py)
        self.non_rigid_mlp = load_non_rigid_motion_mlp(cfg.non_rigid_motion_mlp.module)(
            pos_embed_size=non_rigid_pos_embed_size,
            condition_code_size=cfg.non_rigid_motion_mlp.condition_code_size,
            mlp_width=cfg.non_rigid_motion_mlp.mlp_width,
            mlp_depth=cfg.non_rigid_motion_mlp.mlp_depth,
            skips=cfg.non_rigid_motion_mlp.skips
        )

        self.non_rigid_mlp = nn.DataParallel(
            self.non_rigid_mlp,
            device_ids=cfg.secondary_gpus,
            output_device=cfg.secondary_gpus[0]
        )

        # Canonical MLP -------------------------------------------------------
        # canonical positional encoding
        get_embedder = load_positional_embedder(cfg.embedder.module)   # dj
        cnl_pos_embed_fn, cnl_pos_embed_size = get_embedder(cfg.canonical_mlp.multires, cfg.canonical_mlp.i_embed)
        self.pos_embed_fn = cnl_pos_embed_fn
        
        self.pos_hash_fn, self.in_dim = get_encoder("hashgrid")
        self.dir_sphar_fn, self.in_dim_dir = get_encoder("sphere_harmonics")

        if cfg.network.apply_hash_coding:           
            skips = None # dj: should smaller than cfg.canonical_mlp.mlp_depth #################
            self.cnl_mlp = load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cfg.canonical_mlp.cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips
            )            
        else:
            skips = None # dj: should smaller than cfg.canonical_mlp.mlp_depth #################
            self.cnl_mlp = load_canonical_mlp(cfg.canonical_mlp.module)(
                input_ch=cnl_pos_embed_size, 
                mlp_depth=cfg.canonical_mlp.mlp_depth, 
                mlp_width=cfg.canonical_mlp.mlp_width,
                skips=skips
            )
        
        self.cnl_mlp = nn.DataParallel(
            self.cnl_mlp,
            device_ids=cfg.secondary_gpus,
            output_device=cfg.primary_gpus[0]
        )

        # pose correction -------------------------------------------
        # load_pose_decoder: (core/nets/human_nerf/pose_decoders/mlp_delta_body_pose.py)
        self.pose_decoder = load_pose_decoder(cfg.pose_decoder.module)(
            embedding_size=cfg.pose_decoder.embedding_size,
            mlp_width=cfg.pose_decoder.mlp_width,
            mlp_depth=cfg.pose_decoder.mlp_depth
        )   
        
        scale=1
        #rgb_act='Sigmoid'
        rgb_act='None'
        self.rgb_act = rgb_act
        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16; 
        b = np.exp(np.log(2048*scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = tcnn.NetworkWithInputEncoding(
            n_input_dims=3, n_output_dims=32,
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
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )

        self.dir_encoder = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "SphericalHarmonics",
                "degree": 4,
            },
        )

        self.rgb_net = tcnn.Network(
            n_input_dims=32, n_output_dims=3,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": self.rgb_act,
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )

        #if self.rgb_act == 'None': # rgb_net output is log-radiance
        #    for i in range(3): # independent tonemappers for r,g,b
        #        tonemapper_net = tcnn.Network(
        #            n_input_dims=1, n_output_dims=1,
        #            network_config={
        #                "otype": "FullyFusedMLP",
        #                "activation": "ReLU",
        #                "output_activation": "Sigmoid",
        #                "n_neurons": 64,
        #                "n_hidden_layers": 1,
        #            }
        #        )
        #        setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def deploy_mlps_to_secondary_gpus(self):
        self.cnl_mlp = self.cnl_mlp.to(cfg.secondary_gpus[0])
        if self.non_rigid_mlp:
            self.non_rigid_mlp = self.non_rigid_mlp.to(cfg.secondary_gpus[0])

        return self


    def _query_mlp(
            self,
            pos_xyz,                                                 # dj: [2400, 128, 3]
            pos_dir,                                                 # dj: [2400, 128, 3]
            pos_embed_fn, 
            pos_hash_fn,
            dir_sphar_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input):

        # (N_rays, N_samples, 3) --> (N_rays x N_samples, 3) 
        pos_flat = torch.reshape(pos_xyz, [-1, pos_xyz.shape[-1]])   # dj: [307200, 3]
        dir_flat = torch.reshape(pos_dir, [-1, pos_xyz.shape[-1]])   # dj: [307200, 3]
        chunk = cfg.netchunk_per_gpu*len(cfg.secondary_gpus)         # dj: cfg.netchunk_per_gpu=300000; len(cfg.secondary_gpus)=1

        result = self._apply_mlp_kernels(
            pos_flat=pos_flat,
            dir_flat=dir_flat,
            pos_embed_fn=pos_embed_fn,
            pos_hash_fn=pos_hash_fn,
            dir_sphar_fn=dir_sphar_fn,
            non_rigid_mlp_input=non_rigid_mlp_input,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn,
            chunk=chunk
        )

        output = {}

        raws_flat = result['raws']
        output['raws'] = torch.reshape(raws_flat, list(pos_xyz.shape[:-1]) + [raws_flat.shape[-1]])

        return output


    @staticmethod
    def _expand_input(input_data, total_elem):
        assert input_data.shape[0] == 1
        input_size = input_data.shape[1]
        return input_data.expand((total_elem, input_size))


    def _apply_mlp_kernels(self, pos_flat, dir_flat, pos_embed_fn, pos_hash_fn, dir_sphar_fn, non_rigid_mlp_input, non_rigid_pos_embed_fn, chunk):
        raws = []

        # iterate ray samples by trunks
        for i in range(0, pos_flat.shape[0], chunk): # 307200
            start = i
            end = i + chunk
            if end > pos_flat.shape[0]:
                end = pos_flat.shape[0]
            total_elem = end - start

            xyz = pos_flat[start:end] # dj: [307200, 3] 3D coordinates 
            dir = dir_flat[start:end] # dj: [307200, 3] 3D coordinates' directions
            ### -----------------------------------------
            if not cfg.network.ignore_non_rigid_motions:
                non_rigid_embed_xyz = non_rigid_pos_embed_fn(xyz) # dj: [307200, 36] 36D positional embedding  
                result = self.non_rigid_mlp(
                    pos_embed=non_rigid_embed_xyz,
                    pos_xyz=xyz,
                    condition_code=self._expand_input(non_rigid_mlp_input, total_elem)
                )
                xyz = result['xyz'] # dj: [307200, 3]

            ### original cnl_mlp -------------------------
            if not cfg.network.apply_hash_coding:
                pos_embedded = pos_embed_fn(xyz) # dj: 3D xyz --> 63D pos_embedded
                cnl_mlp_output = self.cnl_mlp(pos_embedded=pos_embedded, dir_embedded=None, hash_encode=cfg.network.apply_hash_coding)
                raws += [cnl_mlp_output]
            ### with hash encoded and tiny mlp -----------
            else: 
                #pos_embedded = pos_hash_fn(xyz) # dj: 3D xyz --> 32D pos_embedded
                #dir_embedded = dir_sphar_fn(dir) # dj: 3D direction --> 16D dir_embedded
                #cnl_mlp_output = self.cnl_mlp(pos_embedded=pos_embedded, dir_embedded=dir_embedded, hash_encode=cfg.network.apply_hash_coding)
                
                # !!! no normalize converge slow
                pos_embedded = self.xyz_encoder(xyz).type(torch.FloatTensor).cuda()

                # !!! sigma converge slow, normalize converge fast
                sigmas, pos_embedded = self.density(xyz, return_feat=True)

                # !!! dir can not converge
                #dir_embedded = dir/torch.norm(dir, dim=1, keepdim=True)
                #dir_embedded = self.dir_encoder((dir_embedded+1)/2).cuda()

                #cnl_mlp_output = self.rgb_net(torch.cat([dir_embedded, pos_embedded], 1))
                #cnl_mlp_output = self.rgb_net(pos_embedded)
                cnl_mlp_output = torch.cat([pos_embedded, sigmas.view(-1, 1)], 1)
                
                #if self.rgb_act == 'None': # rgbs is log-radiance
                #    if kwargs.get('output_radiance', False): # output HDR map
                #        rgbs = TruncExp.apply(rgbs)
                #    else: # convert to LDR using tonemapper networks
                #        rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)
                
                raws += [cnl_mlp_output]


        output = {}
        output['raws'] = torch.cat(raws, dim=0).to(cfg.primary_gpus[0])

        return output


    def _batchify_rays(self, rays_flat, **kwargs):
        all_ret = {}
        for i in range(0, rays_flat.shape[0], cfg.chunk):
            ret = self._render_rays(rays_flat[i:i+cfg.chunk], **kwargs)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])

        all_ret = {k : torch.cat(all_ret[k], 0) for k in all_ret}
        return all_ret


    @staticmethod
    def _raw2outputs(raw, raw_mask, z_vals, rays_d, bgcolor=None):
        def _raw2alpha(raw, dists, act_fn=F.relu):
            return 1.0 - torch.exp(-act_fn(raw)*dists)
            # return 1.0 - torch.exp(-raw*dists)
        # raw: [N_rays, N_samples, 4]
        # raw_mask : [N_rays, N_samples, 1]
        # z_vals : [N_rays, N_samples]
        # rays_d : [N_rays, 3]
        dists = z_vals[..., 1:] - z_vals[..., :-1]

        infinity_dists = torch.Tensor([1e10])
        infinity_dists = infinity_dists.expand(dists[...,:1].shape).to(dists)
        dists = torch.cat([dists, infinity_dists], dim=-1) 
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)                 # [N_rays, N_samples]

        rgb = torch.sigmoid(raw[...,:3])                                       # [N_rays, N_samples, 3]

        # F.relu(raw[...,3]) * dists

        alpha = _raw2alpha(raw[...,3], dists)                                  # [N_rays, N_samples]
        alpha = alpha * raw_mask[:, :, 0]                                      # [N_rays, N_samples]
        weights = alpha * torch.cumprod( torch.cat([torch.ones((alpha.shape[0], 1)).to(alpha), 1.-alpha + 1e-10], dim=-1), dim=-1)[:, :-1]
        rgb_map = torch.sum(weights[...,None] * rgb, -2)                       # [N_rays, 3]

        depth_map = torch.sum(weights * z_vals, -1)                            # [N_rays]
        acc_map = torch.sum(weights, -1)                                       # [N_rays]

        rgb_map = rgb_map + (1.-acc_map[..., None]) * bgcolor[None, :]/255.

        return rgb_map, acc_map, weights, depth_map


    @staticmethod
    def _sample_motion_fields(
            pts,
            motion_scale_Rs, 
            motion_Ts, 
            motion_weights_vol,
            cnl_bbox_min_xyz, 
            cnl_bbox_scale_xyz,
            output_list):
        orig_shape = list(pts.shape)
        pts = pts.reshape(-1, 3) # [N_rays x N_samples, 3]

        # remove BG channel
        motion_weights = motion_weights_vol[:-1] # [24,32,32,32]

        # motion_scale_Rs [24,3,3]; motion_Ts [24,3]
        # cnl_bbox_min_xyz [3]; cnl_bbox_scale_xyz [3]
        weights_list = [] # dj: lenth: #bones, each sample's bones weights
        pos_list = [] ############################ 
        # dj: mapping from observation space to canonical space in bone-wise 
        # from scipy.interpolate import interpn
        # from scipy.interpolate import RegularGridInterpolator
        # import numpy as np
        
        for i in range(motion_weights.size(0)):
            pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :] # dj: pos in cananical space
            pos_list.append(pos) ############################
            pos = (pos - cnl_bbox_min_xyz[None, :]) * cnl_bbox_scale_xyz[None, :] - 1.0 
            
            # print(motion_weights[2,30,:])
            # print(pos[40:50,:])
            # weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], grid=pos[None, None, None, :, :], padding_mode='zeros', align_corners=True)
            weights = F.grid_sample(input=motion_weights[None, i:i+1, :, :, :], grid=pos[None, None, None, :, :], mode='bilinear', padding_mode='zeros', align_corners=True)
            # weights = torch.zeros(1,1,1,1,pos.shape[0]).to(pos.device) ########
            weights = weights[0, 0, 0, 0, :, None] # [1, 1, 1, 1, 307200] to [307200,1]

            # points = (np.linspace(0, 31, 32), np.linspace(0, 31, 32), np.linspace(0, 31, 32))
            # values = motion_weights[i, :].cpu().detach().numpy()
            # shift_pos = (pos + 1.0)*16
            # querys = shift_pos.cpu().detach().numpy()
            # weights = interpn(points, values, querys, method='linear', bounds_error=False, fill_value=0).astype('float32')
            # weights = torch.tensor(weights.reshape(-1,1)).to(pos.device)
   
            weights_list.append(weights) # per canonical pixel's bones weights

        backwarp_motion_weights = torch.cat(weights_list, dim=-1) # dj: [N_rays x N_samples, #bones]
        total_bases = backwarp_motion_weights.shape[-1] # #bones
        backwarp_motion_weights_sum = torch.sum(backwarp_motion_weights, dim=-1, keepdim=True) # dj: [N_rays x N_samples, 1]

        weighted_motion_fields = []
        for i in range(total_bases):
            # pos = torch.matmul(motion_scale_Rs[i, :, :], pts.T).T + motion_Ts[i, :] # dj: pos in cananical space ##################################
            pos = pos_list[i] ##################################
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


    @staticmethod
    def _unpack_ray_batch(ray_batch):
        rays_o, rays_d = ray_batch[:,0:3], ray_batch[:, 3:6] 
        bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2]) 
        near, far = bounds[..., 0], bounds[..., 1] 
        return rays_o, rays_d, near, far


    @staticmethod
    def _get_samples_along_ray(N_rays, near, far):
        t_vals = torch.linspace(0., 1., steps=cfg.N_samples).to(near)
        z_vals = near * (1.-t_vals) + far * (t_vals)
        return z_vals.expand([N_rays, cfg.N_samples]) 


    @staticmethod
    def _stratified_sampling(z_vals):
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape).to(z_vals)
        z_vals = lower + (upper - lower) * t_rand
        return z_vals


    def _render_rays(
            self, 
            ray_batch, 
            motion_scale_Rs,
            motion_Ts,
            motion_weights_vol,
            cnl_bbox_min_xyz,
            cnl_bbox_scale_xyz,
            pos_embed_fn,
            pos_hash_fn,
            dir_sphar_fn,
            non_rigid_pos_embed_fn,
            non_rigid_mlp_input=None,
            bgcolor=None,
            **_):
        
        N_rays = ray_batch.shape[0]
        rays_o, rays_d, near, far = self._unpack_ray_batch(ray_batch)

        z_vals = self._get_samples_along_ray(N_rays, near, far) # dj: [N_rays=2400, nSamples=128]
        
        if cfg.perturb > 0.:
            z_vals = self._stratified_sampling(z_vals)          # dj: [N_rays=2400,nSamples=128]

        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None] # dj: [2400,128,3] (N_rays, N_samples, xyz)

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
        # distloss = self._distortation_loss(cnl_pts,)

        pts_dir = rays_d[..., None, :] * torch.ones_like(z_vals[..., :, None]) # dj: pts' direction (humannerf does not use such info)
        # cnl_pts.shape [2400, 128, 3]; non_rigid_mlp_input [1,69]

        query_result = self._query_mlp(
            pos_xyz=cnl_pts,
            pos_dir=pts_dir,
            non_rigid_mlp_input=non_rigid_mlp_input,
            pos_embed_fn=pos_embed_fn,
            pos_hash_fn=pos_hash_fn,
            dir_sphar_fn=dir_sphar_fn,
            non_rigid_pos_embed_fn=non_rigid_pos_embed_fn
        )
        raw = query_result['raws']
        
        rgb_map, acc_map, _, depth_map = self._raw2outputs(raw, pts_mask, z_vals, rays_d, bgcolor)

        return {'rgb' : rgb_map, 'alpha' : acc_map, 'depth': depth_map}


    def _get_motion_base(self, dst_Rs, dst_Ts, cnl_gtfms):
        motion_scale_Rs, motion_Ts = self.motion_basis_computer(dst_Rs, dst_Ts, cnl_gtfms)

        return motion_scale_Rs, motion_Ts


    @staticmethod
    def _multiply_corrected_Rs(Rs, correct_Rs):
        total_bones = cfg.total_bones - 1
        return torch.matmul(Rs.reshape(-1, 3, 3), correct_Rs.reshape(-1, 3, 3)).reshape(-1, total_bones, 3, 3)

    
    def forward(self, rays, dst_Rs, dst_Ts, cnl_gtfms, motion_weights_priors, motionCLIP, dst_posevec=None, near=None, far=None, iter_val=1e7, **kwargs):
        dst_Rs=dst_Rs[None, ...] # [1, 24, 3, 3]
        dst_Ts=dst_Ts[None, ...] # [1, 24, 3]
        dst_posevec=dst_posevec[None, ...] # [1, 69]
        cnl_gtfms=cnl_gtfms[None, ...]
        motion_weights_priors=motion_weights_priors[None, ...]
        #motionCLIP=motionCLIP[None, ...] # dj [540, 512]

        # correct body pose
        ### -----------------------------------------
        if not cfg.network.ignore_pose_correction: # dj
            if iter_val >= cfg.pose_decoder.get('kick_in_iter', 0):
                pose_out = self.pose_decoder(dst_posevec) # [1, 23, 3, 3] axis-angle (3) to rotation matrix (3,3)
                delta_Rs = pose_out['Rs'] # [1, 23, 3, 3]
                delta_Ts = pose_out.get('Ts', None)
                
                dst_Rs_no_root = dst_Rs[:, 1:, ...]
                dst_Rs_no_root = self._multiply_corrected_Rs(dst_Rs_no_root,delta_Rs)
                dst_Rs = torch.cat([dst_Rs[:, 0:1, ...], dst_Rs_no_root], dim=1)

                if delta_Ts is not None:
                    dst_Ts = dst_Ts + delta_Ts

        # prepare non-rigid motion needed embedding
        ### -----------------------------------------
        
        non_rigid_pos_embed_fn, _ = self.get_non_rigid_embedder(multires=cfg.non_rigid_motion_mlp.multires, is_identity=cfg.non_rigid_motion_mlp.i_embed, iter_val=iter_val)

        # delayed optimization
        if iter_val < cfg.non_rigid_motion_mlp.kick_in_iter:
            # mask-out non_rigid_mlp_input 
            non_rigid_mlp_input = torch.zeros_like(dst_posevec) * dst_posevec
        else:
            non_rigid_mlp_input = dst_posevec

        kwargs.update({
            "pos_embed_fn": self.pos_embed_fn,
            "pos_hash_fn": self.pos_hash_fn,
            "dir_sphar_fn": self.dir_sphar_fn,
            "non_rigid_pos_embed_fn": non_rigid_pos_embed_fn,
            "non_rigid_mlp_input": non_rigid_mlp_input
        })


        # skeletal motion and non-rigid motion
        ### -----------------------------------------
        # dj: dst_Rs [1, 24, 3, 3]; dst_Ts [1, 24, 3], cnl_gtfms [1, 24, 4, 4]
        # dj: motion_scale_Rs [1, 24, 3, 3]; motion_Ts [1, 24, 3] MAPPING from Target pose to T-pose
        motion_scale_Rs, motion_Ts = self._get_motion_base(dst_Rs=dst_Rs, dst_Ts=dst_Ts, cnl_gtfms=cnl_gtfms)
        # dj: motion_weights_vol [25, 32, 32, 32] each bone-level (with BG) weights in volumn (x,y,z)
        motion_weights_vol = self.mweight_vol_decoder(motion_weights_priors=motion_weights_priors)
        motion_weights_vol = motion_weights_vol[0] # remove batch dimension motion_weights_vol [25, 32, 32, 32]

        kwargs.update({
            'motion_scale_Rs': motion_scale_Rs,
            'motion_Ts': motion_Ts,
            'motion_weights_vol': motion_weights_vol
        })


        ### -----------------------------------------
        rays_o, rays_d = rays
        rays_shape = rays_d.shape # [2400, 3]

        rays_o = torch.reshape(rays_o, [-1, 3]).float()
        rays_d = torch.reshape(rays_d, [-1, 3]).float()
        packed_ray_infos = torch.cat([rays_o, rays_d, near, far], -1)

        all_ret = self._batchify_rays(packed_ray_infos, **kwargs) # dj: includs the canonical mlp

        for k in all_ret:
            k_shape = list(rays_shape[:-1]) + list(all_ret[k].shape[1:])
            all_ret[k] = torch.reshape(all_ret[k], k_shape)

        return all_ret

    def density(self, x, return_feat=False):
            """
            Inputs:
                x: (N, 3) xyz in [-scale, scale]
                return_feat: whether to return intermediate feature

            Outputs:
                sigmas: (N)
            """
            x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
            h = self.xyz_encoder(x)
            sigmas = TruncExp.apply(h[:, 0])
            if return_feat: 
                return sigmas, h
            return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs