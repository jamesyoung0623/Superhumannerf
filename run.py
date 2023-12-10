import os

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

import hydra

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask']


def load_network(cfg):
    model = create_network(cfg)

    ckpt_path = os.path.join(cfg.logdir, f'{cfg.load_net}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda()


def unpack_alpha_map(alpha_vals, ray_mask, width, height):
    alpha_map = np.zeros((height * width), dtype='float32')
    alpha_map[ray_mask] = alpha_vals
    return alpha_map.reshape((height, width))


def unpack_to_image(width, height, ray_mask, bgcolor, rgb, alpha, truth=None):
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


def _freeview(cfg, data_type='freeview', folder_name=None):
    cfg.perturb = 0.

    model = load_network(cfg)
    test_loader = create_dataloader(cfg, data_type)
    writer = ImageWriter(output_dir=os.path.join(cfg.logdir, cfg.load_net), exp_name=folder_name)

    model.eval()
    for batch in tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)

        with torch.no_grad():
            net_output, _ = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['weights_sum']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']
        target_rgbs = batch.get('target_rgbs', None)

        rgb_img, alpha_img, _ = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy()
        )

        imgs = [rgb_img]
        imgs.append(alpha_img)

        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out)

    writer.finalize()


def run_freeview(cfg):
    _freeview(cfg, data_type='freeview', folder_name=f"freeview_{cfg.freeview.frame_idx}")


def run_tpose(cfg):
    cfg.network.ignore_non_rigid_motions = True
    _freeview(cfg, data_type='tpose', folder_name='tpose')


def run_movement(cfg, render_folder_name='movement'):
    cfg.perturb = 0.

    model = load_network(cfg)
    test_loader = create_dataloader(cfg, 'movement')
    writer = ImageWriter(output_dir=os.path.join(cfg.logdir, cfg.load_net), exp_name=render_folder_name)

    model.eval()
    for idx, batch in enumerate(tqdm(test_loader)):
        for k, v in batch.items():
            batch[k] = v[0]

        data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])

        with torch.no_grad():
            net_output, _ = model(**data, iter_val=cfg.eval_iter)

        rgb = net_output['rgb']
        alpha = net_output['weights_sum']

        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        rgb_img, alpha_img, truth_img = unpack_to_image(
            width, height, ray_mask, np.array(cfg.bgcolor)/255.,
            rgb.data.cpu().numpy(),
            alpha.data.cpu().numpy(),
            batch['target_rgbs']
        )

        imgs = [rgb_img]
        imgs.append(truth_img)
        imgs.append(alpha_img)
            
        img_out = np.concatenate(imgs, axis=1)
        writer.append(img_out, img_name=f"{idx:06d}")
    
    writer.finalize()

# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/313/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/315/', config_name='single_gpu_hash')
@hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/377/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/386/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/387/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/390/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/392/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/393/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/394/', config_name='single_gpu_hash')
def main(cfg):
    run_movement(cfg)
    run_tpose(cfg)
    run_freeview(cfg)

if __name__ == '__main__':
    main()
