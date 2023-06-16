import os
import sys
import cv2
from shutil import copyfile

import pickle
import yaml
import numpy as np
from tqdm import tqdm
import h5py
# import tqdm

from pathlib import Path
sys.path.append(str(Path(os.getcwd()).resolve().parents[1]))

from third_parties.smpl.smpl_numpy import SMPL
from core.utils.file_util import split_path
from core.utils.image_util import load_image, save_image, to_3ch_image

# from tools.snapshot_smpl.smpl import Smpl  #########

from absl import app
from absl import flags
FLAGS = flags.FLAGS

flags.DEFINE_string('cfg',
                    'male-3-casual.yaml',
                    'the path of config file')

MODEL_DIR = '../../third_parties/smpl/models'

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def parse_config():
    config = None
    with open(FLAGS.cfg, 'r') as file:
        config = yaml.full_load(file)
    return config

def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main(argv):
    del argv  # Unused.

    cfg = parse_config()
    subject = cfg['dataset']['subject']
    sex = cfg['dataset']['sex']
    max_frames = cfg['max_frames']

    dataset_dir = cfg['dataset']['people_snapshot_path']
    subject_dir = os.path.join(dataset_dir, subject)
    output_path = os.path.join(cfg['output']['dir'], subject if 'name' not in cfg['output'].keys() else cfg['output']['name'])
    out_img_dir = prepare_dir(output_path, 'images')
    out_msk_dir = prepare_dir(output_path, 'masks')
    # print('processing ' + video_path + '  to ' + out_img_dir)      
    # print('processing ' + mask_path  + '         to ' + out_msk_dir)     
    # smpl_params_dir = os.path.join(subject_dir, "reconstructed_poses.hdf5")   # smpl parameters
    # select_view = cfg['training_view']
    # keypoints   = h5py.File(os.path.join(subject_dir, 'keypoints.hdf5')) 
    # kpts  = keypoints['keypoints']          # (frames,54)
 

    camera = read_pickle(os.path.join(subject_dir, 'camera.pkl'))
    # K: cam_intrinsics
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1            # (3,3)
    # print(K)
    # E: cam_extrinsics
    R = np.eye(3)          # dummy
    T = np.zeros([3])    # dummy
    D = camera['camera_k'] # (5,)
    E = np.eye(4)  #(4, 4)
    E[:3, :3] = R
    E[:3, 3]= T[:]
#   tmp = read_pickle('../../../People_Snapshot/male-3-casual/consensus.pkl')
    breakpoint()


    # extracting frames and masks
    video_path  = os.path.join(subject_dir, subject + '.mp4')
    masks       = h5py.File(os.path.join(subject_dir, 'masks.hdf5'))['masks']   
    smpl_params = h5py.File(os.path.join(subject_dir, 'reconstructed_poses.hdf5'))

    betas = smpl_params['betas']            # (10,)
    poses  = smpl_params['pose']             # (frames+1,72)
    # trans = smpl_params['trans']            # (frames+1,3)
    poses  = poses[len(poses) - len(masks):]   # 
    # trans = trans[len(trans) - len(masks):] # 
    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR) # dj
    # breakpoint()

    cameras = {}                                    # dj
    mesh_infos = {}                                 # dj
    all_betas = []                                  # dj
    # if 'female' in subject:                    
    #     model_path = '../basicModel_f_lbs_10_207_0_v1.0.0.pkl'
    # else:
    #     model_path = '../basicmodel_m_lbs_10_207_0_v1.0.0.pkl'
    # model_data = read_pickle(model_path)
    # base_smpl = Smpl(model_data)  # (6890,3)
    # vertices, mesh = get_smpl(base_smpl, betas, pose[i], trans[i])
    # base_smpl.betas = betas
    # base_smpl.pose = poses[i]
    # base_smpl.trans = trans[i]
    # vertices = np.array(base_smpl)
    # breakpoint() 
    # betas = smpl_model.beta
    if max_frames>len(masks):
        max_frames = len(masks)
    cap = cv2.VideoCapture(video_path)    
    for i in tqdm(range(max_frames)):    
        out_name = 'frame_{:04d}'.format(i)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(out_img_dir, out_name + '.png'), frame) # dj
        mask = masks[i].astype(np.uint8)
        mask = cv2.erode(mask.copy(), np.ones((4, 4), np.uint8)) * 255
        cv2.imwrite(os.path.join(out_msk_dir, out_name + '.png'), mask) # dj  



        
        all_betas.append(betas[:])

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, tpose_joints = smpl_model(np.zeros_like(poses[i]), betas[:])


        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
        Th = pelvis_pos
        Rh = poses[i][:3].copy()

        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # remove global rotation from body pose
        poses[i][:3] = 0

        # get posed joints using body poses without global rotation
        _, joints = smpl_model(poses[i], betas[:])
        joints = joints - pelvis_pos[None, :]

        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'poses': poses[i],  # (72,)
            'joints': joints, # (24,3)
            'tpose_joints': tpose_joints # (24,3)
        }

        # write camera info
        cameras[out_name] = {
                'intrinsics': K,
                'extrinsics': E,
                'distortions': D
        }


 
    cap.release()
    # breakpoint() 
    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    # smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump({'joints': template_joints,}, f)


    # copy config file
    copyfile(FLAGS.cfg, os.path.join(output_path, 'config.yaml'))



    # breakpoint()



    
    # smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    # cameras = {}
    # mesh_infos = {}
    # all_betas = []
    # for idx, ipath in enumerate(tqdm(img_paths)):
    #     # out_name = 'frame_{:06d}'.format(idx)

    #     # img_path = os.path.join(subject_dir, ipath)
    
    #     # load image
    #     # img = np.array(load_image(img_path))

    #     # if subject in ['313', '315']:
    #     #     _, image_basename, _ = split_path(img_path)
    #     #     start = image_basename.find(')_')
    #     #     smpl_idx = int(image_basename[start+2: start+6])
    #     # else:
    #     #     smpl_idx = idx

    #     # # load smpl parameters
    #     # smpl_params = np.load(
    #     #     os.path.join(smpl_params_dir, f"{smpl_idx}.npy"),
    #     #     allow_pickle=True).item()

    #     betas = smpl_params['shapes'][0] #(10,)
    #     poses = smpl_params['poses'][0]  #(72,)
    #     Rh = smpl_params['Rh'][0]  #(3,)
    #     Th = smpl_params['Th'][0]  #(3,)
        
    #     all_betas.append(betas)

    #     # # write camera info
    #     # cameras[out_name] = {
    #     #         'intrinsics': K,
    #     #         'extrinsics': E,
    #     #         'distortions': D
    #     # }

    #     # write mesh info
    #     _, tpose_joints = smpl_model(np.zeros_like(poses), betas)
    #     _, joints = smpl_model(poses, betas)
    #     mesh_infos[out_name] = {
    #         'Rh': Rh,
    #         'Th': Th,
    #         'poses': poses,
    #         'joints': joints, 
    #         'tpose_joints': tpose_joints
    #     }

    #     # load and write mask
    #     mask = get_mask(subject_dir, ipath)
    #     save_image(to_3ch_image(mask), 
    #                os.path.join(out_mask_dir, out_name+'.png'))

    #     # write image
    #     out_image_path = os.path.join(out_img_dir, '{}.png'.format(out_name))
    #     save_image(img, out_image_path)

    # # write camera infos
    # with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
    #     pickle.dump(cameras, f)
    # # with open(os.path.join(output_path, 'camera.pkl'), 'wb') as f:   # dj
    # #     pickle.dump(cameras, f)

    # # write mesh infos
    # with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
    #     pickle.dump(mesh_infos, f)

    # # write canonical joints
    # avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    # smpl_model = SMPL(sex, model_dir=MODEL_DIR)
    # _, template_joints = smpl_model(np.zeros(72), avg_betas)
    # with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
    #     pickle.dump({'joints': template_joints,}, f)

if __name__ == '__main__':
    app.run(main)
