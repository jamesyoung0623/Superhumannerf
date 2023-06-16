import os

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg, args

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask']
cfg.bgcolor = [200,200,200] # dj 
# cfg.show_alpha = True # dj

def load_network():
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor,
                    rgb, alpha, truth=None):
    
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    alpha_map = unpack_alpha_map(alpha, ray_mask, width, height)
    alpha_image  = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image


def _freeview(
        data_type='freeview',
        folder_name=None):
    cfg.perturb = 0.

    model = load_network()
    test_loader = create_dataloader(data_type)
    writer = ImageWriter(
                output_dir=os.path.join(cfg.logdir, cfg.load_net),
                exp_name=folder_name)



    from core.utils.body_util import body_pose_to_body_RTs # dj: for calling func body_pose_to_body_RTs
    model.eval()
    # for batch in tqdm(test_loader):
    for index, batch in enumerate(tqdm(test_loader)):  # dj   
        for k, v in batch.items():
            batch[k] = v[0]
        # print(index)    
        ##############################################################################
        # breakpoint()
        motion_offset_path = '../MotionCLIP/toHumanNeRF/clip_text_toy_texts_fig_100.pkl'
        motion_offset_motionCLIP = torch.load(motion_offset_path, map_location='cuda:0').cpu().numpy() 
        motion_offset_motionCLIP = motion_offset_motionCLIP.reshape(-1,3,60) # [24,3,60]
        # invert axis
        motion_offset_motionCLIP[:, 1, :] = -motion_offset_motionCLIP[:, 1, :]
        motion_offset_motionCLIP[:, 2, :] = -motion_offset_motionCLIP[:, 2, :]
        # motion_can_path = '../MotionCLIP/exps/paper-model/clip_text_toy_texts_fig_100_can.pkl'
        # motion_can_motionCLIP = torch.load(motion_can_path, map_location='cuda:0').cpu().numpy()         
        # motion_can_motionCLIP = motion_can_motionCLIP.reshape(-1,3,60)


        # breakpoint()
        motion_path = '../MotionCLIP/toHumanNeRF/clip_text_toy_texts_fig_100.pkl'
        motion_axis_angles = torch.load(motion_path, map_location='cuda:0').cpu().numpy() 
        motion_axis_angles = motion_axis_angles.reshape(-1,60) # [72,60]        



        # print('$$$ &&& ---------------')
        # print(batch.keys()) # dict_keys(['img_width', 'img_height', 'ray_mask', 'rays', 'near', 'far', 'bgcolor', 
                              #    'dst_Rs', 'dst_Ts', 'cnl_gtfms', 'motion_weights_priors', 'cnl_bbox_min_xyz', 
                              #    'cnl_bbox_max_xyz', 'cnl_bbox_scale_xyz', 'dst_posevec'])  
                              # dj: add dst_poses and ini_skel_joints for body_pose_to_body_RTs()
        # dj: original pose
        # print(batch['dst_Rs'].shape)                       # dj: [24,3,3] rotation
        # print(batch['dst_Ts'].shape)                       # dj: [24,3]   translation
        # print(batch['dst_posevec'].shape)                  # dj: [69]

        # dj: pose from motionCLIP (see core.data.human_nerf.tpose.py)
        # print(batch['dst_poses'].shape)                    # dj: [72]
        # print(batch['ini_skel_joints'].shape)              # dj: [24,3]
        rel_poses = batch['dst_poses'].numpy()             # dj: should be numpy array
        # breakpoint()
        # rel_poses[5*3+2] = -1.0 # dj: left hip  rotate +1/pi*180 = +57 deg
        # dj: joint_id*3+{0,1,2}, 0:yz; 1:xz, 2: xy
        # rel_poses[3:] = motion_offset_motionCLIP[:,index][3:]
        # rel_poses = motion_offset_motionCLIP[:,index]

        ##################################
        # rel_poses = motion_axis_angles[:,index]
        rel_poses[3:] = motion_axis_angles[:,index][3:]
        ################################

        ini_skel_joints = batch['dst_skel_joints'].numpy() # dj: should be numpy array
        # ini_skel_joints[4,2] += 0.3 # dj   [24,3]
        # offset = [-0.00170379, -0.22081675,  0.02813518]


        # rel_poses[1*3+2] = 0.03 * index
        # rel_poses[4*3+2] = -0.02 * index
        # rel_poses[7*3+2] = -0.01 * index
        # # rel_poses[10*3+2] = -0.02 * index

        # rel_poses[2*3+2] = -0.03 * index
        # rel_poses[5*3+2] = 0.02 * index
        # rel_poses[8*3+2] = 0.01 * index
        # # rel_poses[11*3+2] = -0.02 * index

        # rel_poses[16*3+2] = 0.03 * index
        # rel_poses[18*3+2] = -0.02 * index
        # rel_poses[20*3+2] = -0.01 * index
        # # rel_poses[22*3+2] = -0.02 * index        

        # rel_poses[17*3+2] = -0.03 * index
        # rel_poses[19*3+2] = 0.02 * index
        # rel_poses[21*3+2] = 0.01 * index
        # # rel_poses[23*3+2] = -0.02 * index


        # offset = ini_skel_joints[0,:] - motion_offset_motionCLIP[0,:,index] # offset between different root joints
        # ini_skel_joints = motion_offset_motionCLIP[:,:,index] + offset # viewing in 360 degrees
        breakpoint()
        

        dst_Rs, dst_Ts = body_pose_to_body_RTs(rel_poses, ini_skel_joints) # inputs are numpy arrays
        batch['dst_Rs'] = torch.tensor(dst_Rs)             # dj: should be tensor
        batch['dst_Ts'] = torch.tensor(dst_Ts)             # dj: should be tensor
        # breakpoint()
        ##############################################################################



        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy())

        imgs = [rgb_img]
        if cfg.show_truth and target_rgbs is not None:
            target_rgbs = to_8b_image(target_rgbs.numpy())
            imgs.append(target_rgbs)
        if cfg.show_alpha:
            imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out)

    writer.finalize()


def run_freeview():
    _freeview(
        data_type='freeview',
        folder_name=f"freeview_{cfg.freeview.frame_idx}" \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_tpose():
    cfg.ignore_non_rigid_motions = True
    _freeview(
        data_type='tpose',
        folder_name='tpose' \
            if not cfg.render_folder_name else cfg.render_folder_name)


def run_movement(render_folder_name='movement'):
    cfg.perturb = 0.

    model = load_network()
    # breakpoint()
  
    test_loader = create_dataloader('movement')
    writer = ImageWriter(
        output_dir=os.path.join(cfg.logdir, cfg.load_net),
        exp_name=render_folder_name)

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        print('$$$ &&& ---------------')
        # print(batch.keys()) # dict_keys(['img_width', 'img_height', 'ray_mask', 'rays', 'near', 'far', 'bgcolor', 
                                #    'dst_Rs', 'dst_Ts', 'cnl_gtfms', 'motion_weights_priors', 'cnl_bbox_min_xyz', 
                                #    'cnl_bbox_max_xyz', 'cnl_bbox_scale_xyz', 'dst_posevec'])  
        print(batch['dst_Rs'].shape) # dj: [24,3,3]
        print(batch['dst_Ts'].shape) # dj: [24,3] 
        print(batch['dst_posevec'].shape) # dj: [69]
        # print(batch['dst_posevec'][0:68])   
        batch['dst_posevec'][:] = 0.01

        data = cpu_data_to_gpu(
                    batch,
                    exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output = model(**data, iter_val=cfg.eval_iter)
        # breakpoint()


        rgb = net_output['rgb']
        alpha = net_output['alpha']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = \
            unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor)/255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs'])

        imgs = [rgb_img]
        if cfg.show_truth:
            imgs.append(truth_img)
        if cfg.show_alpha:
            imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=f"{idx:06d}")
    
    writer.finalize()

        
if __name__ == '__main__':
    globals()[f'run_{args.type}']()
