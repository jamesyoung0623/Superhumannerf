import os
import sys
import cv2

import pickle
import yaml
import numpy as np
from tqdm import tqdm
import h5py

sys.path.append('./')

from third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = './third_parties/smpl/models'

def read_pickle(pkl_path):
    with open(pkl_path, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        return u.load()

def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main():
    subject = 'male-3-casual'
    sex = 'neutral'

    dataset_dir = '../People_Snapshot'
    subject_dir = os.path.join(dataset_dir, subject)
    output_path = os.path.join('../datasets/people_snapshot', subject)
    out_img_dir = prepare_dir(output_path, 'images')
    out_msk_dir = prepare_dir(output_path, 'masks')

    camera = read_pickle(os.path.join(subject_dir, 'camera.pkl'))
    # K: cam_intrinsics
    K = np.zeros([3, 3])
    K[0, 0] = camera['camera_f'][0]
    K[1, 1] = camera['camera_f'][1]
    K[:2, 2] = camera['camera_c']
    K[2, 2] = 1

    R = np.eye(3)
    T = np.zeros([3])
    D = camera['camera_k']
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3]= T[:]

    # extracting frames and masks
    video_path  = os.path.join(subject_dir, subject + '.mp4')
    masks       = h5py.File(os.path.join(subject_dir, 'masks.hdf5'))['masks']   
    smpl_params = h5py.File(os.path.join(subject_dir, 'reconstructed_poses.hdf5'))

    betas = smpl_params['betas']            # (10,)
    poses  = smpl_params['pose']             # (frames+1,72)
    poses  = poses[len(poses) - len(masks):]

    max_frames = poses.shape[0]

    smpl_model = SMPL(sex=sex, model_dir=MODEL_DIR)

    cameras = {}                                    # dj
    mesh_infos = {}                                 # dj
    all_betas = []                                  # dj

    cap = cv2.VideoCapture(video_path)    
    for i in tqdm(range(max_frames)):    
        out_name = 'frame_{:04d}'.format(i)
        ret, frame = cap.read()
        cv2.imwrite(os.path.join(out_img_dir, out_name + '.png'), frame)
        mask = masks[i].astype(np.uint8)
        mask = cv2.erode(mask.copy(), np.ones((4, 4), np.uint8)) * 255
        cv2.imwrite(os.path.join(out_msk_dir, out_name + '.png'), mask)  
        
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

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    avg_betas = np.mean(np.stack(all_betas, axis=0), axis=0)
    _, template_joints = smpl_model(np.zeros(72), avg_betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump({'joints': template_joints,}, f)

if __name__ == '__main__':
    main()
