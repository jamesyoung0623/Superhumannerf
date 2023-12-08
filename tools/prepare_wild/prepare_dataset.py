import os
import sys
import json
import pickle
import numpy as np
from tqdm import tqdm
import glob

sys.path.append('./')

from third_parties.smpl.smpl_numpy import SMPL

MODEL_DIR = './third_parties/smpl/models'

def prepare_dir(output_path, name):
    out_dir = os.path.join(output_path, name)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir

def main():
    subject = 'pitching'
    sex = 'neutral'

    dataset_dir = '../EasyMocap/output/sv1p/smpl/'
    subject_dir = os.path.join(dataset_dir, subject)
    output_path = os.path.join('../datasets/wild', subject)
    out_img_dir = prepare_dir(output_path, 'images')
    out_msk_dir = prepare_dir(output_path, 'masks')

    K = np.array(
        [[1296.0, 0.0, 540.0],
        [0.0, 1296.0, 540.0],
        [0.0, 0.0, 1.0]]
    )

    E = np.eye(4) 
    D = np.array([0.0,  0.0, 0.0, 0.0, 0.0])
    
    smpl_files = sorted(glob.glob(subject_dir+'/*'))

    max_frames = len(smpl_files)

    smpl_model = SMPL(sex='neutral', model_dir='./third_parties/smpl/models') 

    cameras = {}                                   
    mesh_infos = {}  

    for i in tqdm(range(max_frames)): 
        out_name = 'frame_{:04d}'.format(i)
        smpl_file = smpl_files[i]

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        jsonfile = open(smpl_file, newline='')
        smpl = json.load(jsonfile)[0]
        betas = np.array(smpl['shapes'][0])
        pose = np.array([0.0, 0.0, 0.0] + smpl['poses'][0])
        #pose = np.array(smpl['Rh'][0] + smpl['poses'][0])
        Rh = np.array(smpl['Rh'][0])
        Th = np.array(smpl['Th'][0])

        ##############################################
        # Below we tranfer the global body rotation to camera pose

        # Get T-pose joints
        _, tpose_joints = smpl_model(np.zeros_like(pose), betas)

        # get global Rh, Th
        pelvis_pos = tpose_joints[0].copy()
            
        # get refined T-pose joints
        tpose_joints = tpose_joints - pelvis_pos[None, :]

        # get posed joints using body poses without global rotation
        _, joints = smpl_model(pose, betas)
        joints = joints - pelvis_pos[None, :]

        mesh_infos[out_name] = {
            'Rh': Rh,
            'Th': Th,
            'betas': betas,
            'poses': pose,
            'joints': joints,
            'tpose_joints': tpose_joints
        }

        # write camera info
        cameras[out_name] = {
            'intrinsics': K,
            'extrinsics': E,
            'distortions': D
        }

        #plot_joints(joints, tpose_joints)

    # write camera infos
    with open(os.path.join(output_path, 'cameras.pkl'), 'wb') as f:   
        pickle.dump(cameras, f)

    # write mesh infos
    with open(os.path.join(output_path, 'mesh_infos.pkl'), 'wb') as f:   
        pickle.dump(mesh_infos, f)

    # write canonical joints
    _, template_joints = smpl_model(np.zeros(72), betas)
    with open(os.path.join(output_path, 'canonical_joints.pkl'), 'wb') as f:   
        pickle.dump({'joints': template_joints,}, f)

if __name__ == '__main__':
    main()