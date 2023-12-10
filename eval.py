import os

import torch
import numpy as np

from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import to_8b_image

from third_parties.lpips import LPIPS

import hydra

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask']

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_network(cfg, checkpoint):
    model = create_network(cfg)
    ckpt_path = os.path.join(cfg.logdir, f'{checkpoint}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda()


def unpack_to_image(width, height, ray_mask, bgcolor, rgb, truth=None):
    rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
    truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

    rgb_image[ray_mask] = rgb
    rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

    if truth is not None:
        truth_image[ray_mask] = truth
        truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

    return rgb_image, truth_image


def lpips_metric(model, pred, target):
    processed_pred = torch.from_numpy(pred).float().unsqueeze(0).cuda() * 2. - 1.
    processed_target = torch.from_numpy(target).float().unsqueeze(0).cuda() * 2. - 1.

    lpips_loss = model(processed_pred.permute(0, 3, 1, 2), processed_target.permute(0, 3, 1, 2))
    return torch.mean(lpips_loss).cpu().detach().item()


# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/313/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/315/', config_name='single_gpu_hash')
@hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/377/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/386/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/387/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/390/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/392/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/393/', config_name='single_gpu_hash')
# @hydra.main(version_base=None, config_path='configs/human_nerf/zju_mocap/394/', config_name='single_gpu_hash')
def eval_model(cfg):
    cfg.perturb = 0.

    test_loader = create_dataloader(cfg, 'movement')

    lpips_model = LPIPS(net='vgg')
    set_requires_grad(lpips_model, requires_grad=False)
    lpips_model.cuda()

    model = load_network(cfg, 'latest')
    model.eval()
    
    PSNR_list = []
    SSIM_list = []
    LPIPS_list = []

    for batch in tqdm(test_loader):
        for k, v in batch.items():
            batch[k] = v[0]
        width = batch['img_width']
        height = batch['img_height']
        ray_mask = batch['ray_mask']

        # prediction
        data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
        with torch.no_grad():
            net_output, _ = model(**data, iter_val=cfg.eval_iter)
            rgb = net_output['rgb']

            rgb_img, truth_img = unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                batch['target_rgbs']
            )

            rgb_img_norm = rgb_img / 255.
            truth_img_norm = truth_img / 255.

            psnr = compare_psnr(rgb_img_norm, truth_img_norm, data_range=1)
            ssim = compare_ssim(rgb_img_norm, truth_img_norm, data_range=1, channel_axis=2)
            lpips = lpips_metric(model=lpips_model, pred=rgb_img_norm, target=truth_img_norm)

            PSNR_list.append(psnr)
            SSIM_list.append(ssim)
            LPIPS_list.append(lpips)

    psnr_mean = np.mean(PSNR_list).item()
    ssim_mean = np.mean(SSIM_list).item()
    lpips_mean = np.mean(LPIPS_list).item()
    print("PSNR: {:.2f}, SSIM: {:.4f}, LPIPS: {:.2f}".format(psnr_mean, ssim_mean, 1000*lpips_mean))


if __name__ == '__main__':
    eval_model()
