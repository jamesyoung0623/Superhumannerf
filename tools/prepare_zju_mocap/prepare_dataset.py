import os
import sys

from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1])) # '/home/djchen/PROJECTS/HumanNeRF/superhumannerf' 

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

from absl import app
from absl import flags
FLAGS = flags.FLAGS
import torch.nn as nn

flags.DEFINE_string('cfg', '387.yaml', 'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'
 

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)

    return config


def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)

    return out_dir


def get_mask(subject_dir, img_name):
    msk_path = os.path.join(subject_dir, 'mask',
                            img_name)[:-4] + '.png'
    msk = np.array(load_image(msk_path))[:, :, 0]
    msk = (msk != 0).astype(np.uint8)

    msk_path = os.path.join(subject_dir, 'mask_cihp',
                            img_name)[:-4] + '.png'
    msk_cihp = np.array(load_image(msk_path))[:, :, 0]
    msk_cihp = (msk_cihp != 0).astype(np.uint8)

    msk = (msk | msk_cihp).astype(np.uint8)
    msk[msk == 1] = 255

    return msk


# dj: function derived from MotionCLIP --------------------------------------------
# sys.path.insert(1, '/home/djchen/PROJECTS/HumanNeRF/MotionCLIP')
sys.path.append('/home/djchen/PROJECTS/HumanNeRF/MotionCLIP')
import src.utils.rotation_conversions as geometry
from src.visualize.visualize import get_gpu_device
from src.utils.misc import load_model_wo_clip
import src.utils.fixseed  # noqa
import torch
def encode_motions(model, motions, device):
    # z = model.encoder({'x': motions,'y': torch.zeros(motions.shape[0], dtype=int, device=device),'mask': model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=device) * 60)})["mu"]
    # all_tokens = model.encoder({'x': motions,'y': torch.zeros(motions.shape[0], dtype=int, device=device),'mask': model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=device) * 60)})["all"]
    output = model.encoder({'x': motions,'y': torch.zeros(motions.shape[0], dtype=int, device=device),'mask': model.lengths_to_mask(torch.ones(motions.shape[0], dtype=int, device=device) * 60)})
    # breakpoint()
    return output

def get_encode_motions(poses_axis_angle, neighborhood, ZorSelf):
    # neighborhood 1: each frame encodes its clip feature independently
    # neighborhood odds within [1,59]
    # ZorSelf: True for z token; False for token of each frame
    assert neighborhood % 2 != 0 and 1 <= neighborhood <= 59, 'unreasonable neighborhood!!! should be odds within [1,59]'
    
    numPoses = len(poses_axis_angle)
    input_features = torch.zeros(numPoses, 25, 6).float().cuda()
    print('transform each pose from axis_angle to 6d...')  
    dummy_translation = torch.zeros(1, 6).float().cuda()
    for idx, ipose in enumerate(tqdm(poses_axis_angle)):
        # joint-level axis_angle to 6d
        poses_matriz = geometry.axis_angle_to_matrix(torch.Tensor(ipose.reshape(-1,3)).float().cuda())
        poses_6d = geometry.matrix_to_rotation_6d(poses_matriz)
        poses_6d = torch.cat((poses_6d,dummy_translation),0) # [25,6]
        # poses_6d = torch.unsqueeze(poses_6d,3)
        input_features[idx,:] = poses_6d # [nFrames, 25, 6]
    
    
    # print('prepare each pose for feeding MotionCLIP...')
    offset = (neighborhood-1)//2
    if offset != 0:
        padding = torch.zeros(offset, 25, 6).float().cuda()
        input_features = torch.cat((padding,input_features,padding),0) # [25,6]

    # load parameters from MotionCLIP
    parameters = torch.load('/home/djchen/PROJECTS/HumanNeRF/MotionCLIP/toHumanNeRF/parameters.pkl')
    model = torch.load('/home/djchen/PROJECTS/HumanNeRF/MotionCLIP/toHumanNeRF/model.pkl')
    parameters["device"] = f"cuda:{get_gpu_device()}"
    checkpointpath = '/home/djchen/PROJECTS/HumanNeRF/MotionCLIP/exps/paper-model/checkpoint_0100.pth.tar'
    state_dict = torch.load(checkpointpath, map_location=parameters["device"])
    load_model_wo_clip(model, state_dict)


    print('encoding via MotionCLIP...')
    center_idx = 30
    encoded_features = torch.zeros(numPoses, 512).float().cuda()
    for idx, ipose in enumerate(tqdm(poses_axis_angle)):
        container = torch.zeros(60, 25, 6).float().cuda()  # [nframes, njoints(25:translation), nfeats]
        current_idx = idx+offset
        container[center_idx-offset:center_idx+offset+1,:] = input_features[current_idx-offset:current_idx+offset+1]
        container = torch.unsqueeze(container,0)
        container = container.permute((0, 2, 3, 1))
        # breakpoint()
        # see MotionCLIP.src.models.architectures.transformer.py
        motionCLIP_motions = encode_motions(model, container, parameters['device']) # dj: [2,512]
        # breakpoint()
        if ZorSelf:
            encoded_features[idx,:] = motionCLIP_motions['mu']
        else:
            encoded_features[idx,:] = motionCLIP_motions['all'][center_idx-1+2,:]
    # breakpoint()
    return encoded_features
# dj: function derived from MotionCLIP --------------------------------------------


def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['zju_mocap_path']
    subject_dir = os.path.join(dataset_dir, f"CoreView_{subject}")
    smpl_params_dir = os.path.join(subject_dir, "new_params")            # smpl parameters

    anno_path = os.path.join(subject_dir, 'annots.npy')
    annots = np.load(anno_path, allow_pickle=True).item()                # 23 cameras, imaage paths

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)


    ###############################################################################
    # prepare training data
    ###############################################################################
    training_view = cfg['training_view']

    # dj : single camera ------------------------
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[training_view].astype('float32')   ###
    cam_Rs = np.array(cams['R'])[training_view].astype('float32')   ###
    cam_Ts = np.array(cams['T'])[training_view].astype('float32') / 1000. ##
    cam_Ds = np.array(cams['D'])[training_view].astype('float32')

    K = cam_Ks     #(3, 3)  ###############
    D = cam_Ds[:, 0]
    E = np.eye(4)  #(4, 4)  ###########
    cam_T = cam_Ts[:3, 0]   ############
    E[:3, :3] = cam_Rs     ################
    E[:3, 3]= cam_T         ###########
    # dj : single camera ------------------------

    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[training_view] \
            for multi_view_paths in img_path_frames_views
    ])
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    output_path_train = os.path.join(cfg['output']['dir'], 
                               subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    os.makedirs(output_path_train, exist_ok=True)
    out_img_dir  = prepare_dir(output_path_train, 'images')
    out_mask_dir = prepare_dir(output_path_train, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path_train, 'config.yaml'))

    cameras = {}
    mesh_infos = {}
    all_betas = []
    all_poses = []
    for idx, ipath in enumerate(tqdm(img_paths)):
        # if idx>=20:
        #     continue
        out_name = 'frame_{:04d}'.format(idx)

        img_path = os.path.join(subject_dir, ipath)
    
        # load image
        img = np.array(load_image(img_path))

        if subject in ['313', '315']:
            _, image_basename, _ = split_path(img_path)
            start = image_basename.find(')_')
            smpl_idx = int(image_basename[start+2: start+6])
        else:
            smpl_idx = idx

        # load smpl parameters
        smpl_params = np.load(
            os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
            allow_pickle=True).item()

        betas = smpl_params['shapes'][0] #(10,)
        poses = smpl_params['poses'][0]  #(72,)
        Rh = smpl_params['Rh'][0]  #(3,)
        Th = smpl_params['Th'][0]  #(3,)
        
        all_betas.append(betas)

        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
        }

        # write mesh info
        # breakpoint()
        _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
        _, joints = smpl_model(poses, betas)
        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses, # axis-angle vector
            'joints': joints, 
            'tpose_joints': tpose_joints
        }

        # dj: save motionCLIP encoded motions ####################
        # breakpoint()
        all_poses.append(poses)
        

        # load and write mask
        mask = get_mask(subject_dir, ipath)
        save_image(to_3ch_image(mask), 
                   os.path.join(out_mask_dir, out_name+'.png'))

        # write image
        out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
        save_image(img, out_image_path)


    # write camera infos
    with open(os.path.join(output_path_train, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path_train, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path_train, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump(
            {
                'joints': template_joints,
            }, f)


    ###############################################################################
    # prepare evaluating data
    ###############################################################################
    evaluating_view = []
    views = cfg['eval_view'].split(',')
    for view_range in views:
        view_range = view_range.strip()
        index = view_range.find('-')

        if index == -1:
            view_range = int(view_range)
            if view_range < 0 or view_range > 22:
                print(f'eval view invaild! camera index {view_range} is given!')
            else:
                evaluating_view.append(view_range)
        else:
            # view_range is a real range
            for i in range(int(view_range[:index]),int(view_range[index+1:])+1):
                evaluating_view.append(i)
    evaluating_view = np.array(evaluating_view)

    # dj : multiple cameras ------------------------
    # load cameras
    cams = annots['cams']
    cam_Ks = np.array(cams['K'])[evaluating_view].astype('float32')  # view_num*3*3
    cam_Rs = np.array(cams['R'])[evaluating_view].astype('float32')  # view_num*3*3
    cam_Ts = np.array(cams['T'])[evaluating_view].astype('float32') / 1000.  # view_num*3*1
    cam_Ds = np.array(cams['D'])[evaluating_view].astype('float32')  # view_num*5*1

    K = cam_Ks  # view_num*3*3
    D = cam_Ds[..., 0]  # view_num*5
    E = np.zeros((cam_Ks.shape[0], 4, 4)).astype('float32')  # view_num*4*4
    cam_T = cam_Ts[..., 0]
    E[:, :3, :3] = cam_Rs
    E[:, :3, 3] = cam_T
    E[:, 3, 3] = 1.  # view_num*4*4
    # dj : multiple cameras ------------------------

    # load image paths
    img_path_frames_views = annots['ims']
    img_paths = np.array([
        np.array(multi_view_paths['ims'])[evaluating_view] \
        for multi_view_paths in img_path_frames_views
    ])
    img_paths = np.stack(img_paths, 0)  # dj: change 2 dim list to matrix (frame_num, view_num)
    if max_frames > 0:
        img_paths = img_paths[:max_frames]

    # dj: skip some frame ------------------------
    index_keep = np.array(range(0, len(img_paths), cfg['eval_skip']))
    img_paths = img_paths[index_keep]
    # dj: skip some frame ------------------------

    output_path_eval = os.path.join(cfg['eval_output']['dir'],
                               subject if 'name' not in cfg['eval_output'].keys() else cfg['eval_output']['name'])
    os.makedirs(output_path_eval, exist_ok=True)
    out_img_dir = prepare_dir(output_path_eval, 'images')
    out_mask_dir = prepare_dir(output_path_eval, 'masks')

    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path_eval, 'config.yaml'))

    cameras = {}
    mesh_infos = {}
    all_betas = []
    [all_betas.append([]) for i in range(len(img_paths))]             # dj ------------------------
    for idx_frame, path_frame in enumerate(tqdm(img_paths)):          # dj ------------------------
        for idx_camera, path_camera in enumerate(path_frame):         # dj ------------------------
            real_idx_frame = idx_frame * cfg['eval_skip']             # dj ------------------------
            real_idx_camera = evaluating_view[idx_camera].item() + 1  # dj ------------------------
            out_name = 'camera_{:02d}_frame_{:06d}'.format(real_idx_camera, real_idx_frame) # dj ------------------------

            img_path = os.path.join(subject_dir, path_camera)

            # load image
            img = np.array(load_image(img_path))
            
            if subject in ['313', '315']:
                smpl_idx = real_idx_frame + 1  # index begin with 1 # dj ------------------------
            else:
                smpl_idx = real_idx_frame                           # dj ------------------------

            # load smpl parameters
            smpl_params = np.load(
                os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
                allow_pickle=True).item()

            betas = smpl_params['shapes'][0]  # (10,)
            poses = smpl_params['poses'][0]  # (72,)
            Rh = smpl_params['Rh'][0]  # (3,)
            Th = smpl_params['Th'][0]  # (3,)

            all_betas[idx_frame].append(betas)

            # write camera info
            cameras[out_name] = {
                'intrinsics': K[idx_camera], # dj ------------------------
                'extrinsics': E[idx_camera], # dj ------------------------
                'distortions': D[idx_camera] # dj ------------------------
            }

            # write mesh info
            _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
            _, joints = smpl_model(poses, betas)
            mesh_infos[out_name] = {
                'Rh': Rh,
                'Th': Th,
                'poses': poses,
                'joints': joints,
                'tpose_joints': tpose_joints
            }

            # load and write mask
            mask = get_mask(subject_dir, path_camera)
            save_image(to_3ch_image(mask),
                       os.path.join(out_mask_dir, out_name + '.png'))

            # write image
            out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
            save_image(img, out_image_path)

    # write camera infos
    with open(os.path.join(output_path_eval, 'cameras.pkl'), 'wb') as f:
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path_eval, 'mesh_infos.pkl'), 'wb') as f:
        pickle.dump(mesh_infos, f)

    # write canonical joints
    # eliminate duplicate values ​​in all_betas
    all_betas = [all_betas[i][0] for i in range(len(all_betas))] # dj ------------------------
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path_eval, 'canonical_joints.pkl'), 'wb') as f:
        pickle.dump(
            {
                'joints': template_joints,
            }, f)
 




    
    ###############################################################################
    # prepare motionCLIP data
    ###############################################################################    

    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=1, ZorSelf=True)
    with open(os.path.join(output_path_train, 'motionCLIP_token[z]_nbrs[1].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)  
    with open(os.path.join(output_path_eval, 'motionCLIP_token[z]_nbrs[1].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)         
    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=1, ZorSelf=False)
    with open(os.path.join(output_path_train, 'motionCLIP_token[s]_nbrs[1].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f) 
    with open(os.path.join(output_path_eval, 'motionCLIP_token[s]_nbrs[1].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f) 

    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=7, ZorSelf=True)
    with open(os.path.join(output_path_train, 'motionCLIP_token[z]_nbrs[7].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)  
    with open(os.path.join(output_path_eval, 'motionCLIP_token[z]_nbrs[7].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)          
    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=7, ZorSelf=False)
    with open(os.path.join(output_path_train, 'motionCLIP_token[s]_nbrs[7].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)  
    with open(os.path.join(output_path_eval, 'motionCLIP_token[s]_nbrs[7].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)              

    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=15, ZorSelf=True)
    with open(os.path.join(output_path_train, 'motionCLIP_token[z]_nbrs[15].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)  
    with open(os.path.join(output_path_eval, 'motionCLIP_token[z]_nbrs[15].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)          
    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=15, ZorSelf=False)
    with open(os.path.join(output_path_train, 'motionCLIP_token[s]_nbrs[15].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)            
    with open(os.path.join(output_path_eval, 'motionCLIP_token[s]_nbrs[15].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)    

    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=59, ZorSelf=True)
    with open(os.path.join(output_path_train, 'motionCLIP_token[z]_nbrs[59].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)  
    with open(os.path.join(output_path_eval, 'motionCLIP_token[z]_nbrs[59].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)          
    # dj: save motionCLIP encoded motions ####################
    motionCLIP_motions = get_encode_motions(all_poses, neighborhood=59, ZorSelf=False)
    with open(os.path.join(output_path_train, 'motionCLIP_token[s]_nbrs[59].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f)  
    with open(os.path.join(output_path_eval, 'motionCLIP_token[s]_nbrs[59].pkl'), 'wb') as f:   
        pickle.dump(motionCLIP_motions, f) 



  






if __name__ == '__main__':
    app.run(main)
