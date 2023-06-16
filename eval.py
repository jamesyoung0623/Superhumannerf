import os
import skimage

import torch
import numpy as np
from tqdm import tqdm

from core.data import create_dataloader
from core.nets import create_network
from core.utils.train_util import cpu_data_to_gpu
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from configs import cfg
from torch.utils.tensorboard import SummaryWriter

from third_parties.lpips import LPIPS
import glob
import yaml

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask']

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def load_network(cheakpoint):
    model = create_network()
    ckpt_path = os.path.join(cfg.logdir, f'{cheakpoint}.tar')
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    model.load_state_dict(ckpt['network'], strict=False)
    print('load network from ', ckpt_path)
    return model.cuda().deploy_mlps_to_secondary_gpus()

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
    alpha_image = to_8b3ch_image(alpha_map)

    return rgb_image, alpha_image, truth_image

def psnr_metric(img_pred, img_gt):
    ''' Caculate psnr metric
        Args:
            img_pred: ndarray, W*H*3, range 0-1
            img_gt: ndarray, W*H*3, range 0-1

        Returns:
            psnr metric: scalar
    '''
    mse = np.mean((img_pred - img_gt) ** 2)
    psnr = -10 * np.log(mse) / np.log(10)
    return psnr.item()


def lpips_metric(model, pred, target):
    # convert range from 0-1 to -1-1
    processed_pred = torch.from_numpy(pred).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.
    processed_target=torch.from_numpy(target).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.

    lpips_loss = model(processed_pred.permute(0, 3, 1, 2),
                       processed_target.permute(0, 3, 1, 2))
    return torch.mean(lpips_loss).cpu().detach().item()

def save_result_yaml(file_name, result):
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as f:
        yaml.dump(result, f) 


def eval_model(render_folder_name='eval', show_truth=True, show_alpha=True):
    cfg.perturb = 0.
    only_latest = False #######################
    # breakpoint()
    checkpoint_paths = sorted(glob.glob(os.path.join(cfg.logdir, '*.tar')), key=os.path.getmtime)
    checkpoint_num = len(checkpoint_paths)
    # breakpoint()
    # summary folder for tensorboard
    test_loader = create_dataloader('movement')
    log_dir = os.path.join(cfg.logdir, 'logs')
    swriter = SummaryWriter(log_dir)

    # create lpip model and config
    lpips_model = LPIPS(net='vgg')
    set_requires_grad(lpips_model, requires_grad=False)
    lpips_model.to(cfg.primary_gpus[0])


    PSNRD  = {} # all samples
    SSIMD  = {} # all samples
    LPIPSD = {} # all samples
    PSNRM  = {} # all samples' average
    SSIMM  = {} # all samples' average
    LPIPSM = {} # all samples' average
    for n in tqdm(range(checkpoint_num)):
        # if n==5: break #######################
        # evaluation per checkpoint
        path = os.path.splitext(checkpoint_paths[n])
        checkpoint_name = path[0].split('/')[-1]
        # checkpoint_name = cfg.load_net # latest
        if only_latest:
            if checkpoint_name != cfg.load_net:
                continue
        model = load_network(checkpoint_name)
        out_folder_name = checkpoint_name+'_'+render_folder_name # latest_eval
        writer = ImageWriter(output_dir=cfg.logdir, exp_name=out_folder_name) # will create folder

        model.eval()
        PSNRA = []
        SSIMA = []
        LPIPSA = []
        for idx, batch in enumerate(test_loader):
            # if idx==3: break #######################
            # batch data
            for k, v in batch.items():
                batch[k] = v[0]
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            # prediction
            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = model(**data, iter_val=cfg.eval_iter)
            rgb = net_output['rgb']
            alpha = net_output['alpha']

            # *_img: ndarray, (512, 512, 3), value range 0-255
            rgb_img, alpha_img, truth_img = \
                unpack_to_image(
                    width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                    rgb.data.cpu().numpy(),
                    alpha.data.cpu().numpy(),
                    batch['target_rgbs'])
            imgs = [rgb_img]
            if show_truth:
                imgs.append(truth_img)
            if show_alpha:
                imgs.append(alpha_img)
            img_out = np.concatenate(imgs, axis=1)
            writer.append(img_out, img_name=batch['frame_name'])
            # convert image to 0-1
            rgb_img_norm = rgb_img / 255.
            truth_img_norm = truth_img / 255.
            # caculate the metric
            psnr = psnr_metric(rgb_img_norm, truth_img_norm)
            ssim = skimage.metrics.structural_similarity(rgb_img_norm, truth_img_norm, multichannel=True)
            lpips = lpips_metric(model=lpips_model, pred=rgb_img_norm, target=truth_img_norm)
            print(f"Checkpoint [{checkpoint_name}] - {idx}/{len(test_loader)}: PSNR is {psnr:.4f}, SSIM is {ssim:.4f}, LPIPS is {lpips*1000:.4f}")
            # breakpoint()

            swriter.add_scalar('PSNR/'+checkpoint_name, psnr, idx)
            swriter.add_scalar('SSIM/'+checkpoint_name, ssim, idx)
            swriter.add_scalar('LPIPS/'+checkpoint_name, lpips, idx)
            PSNRA.append(psnr)
            SSIMA.append(ssim)
            LPIPSA.append(lpips)
        psnr_mean = np.mean(PSNRA).item()
        ssim_mean = np.mean(SSIMA).item()
        lpips_mean = np.mean(LPIPSA).item()
        print(f"Checkpoint [{checkpoint_name}]: mPSNR is {psnr_mean}, mSSIM is {ssim_mean}, mLPIPS is {lpips_mean*1000}")
        writer.finalize()
        # swriter.add_scalars("Summary-AllCP", {'mPSNR':psnr_mean, 'mSSIM':ssim_mean, 'mLPIPS':lpips_mean*1000},n)
        swriter.add_scalar('AllCP_PSNR', psnr_mean, n)
        swriter.add_scalar('AllCP_SSIM', ssim_mean, n)
        swriter.add_scalar('AllCP_LPIPS', lpips_mean*1000, n)
        PSNRD[checkpoint_name]  = PSNRA
        SSIMD[checkpoint_name]  = SSIMA
        LPIPSD[checkpoint_name] = LPIPSA   
        PSNRM[checkpoint_name]  = psnr_mean
        SSIMM[checkpoint_name]  = ssim_mean
        LPIPSM[checkpoint_name] = lpips_mean*1000             
    save_result_yaml(os.path.join(cfg.result, 'PSNR_all_checkpoints.yaml'), PSNRD)
    save_result_yaml(os.path.join(cfg.result, 'SSIM_all_checkpoints.yaml'), SSIMD)
    save_result_yaml(os.path.join(cfg.result, 'LPIPS_all_checkpoints.yaml'), LPIPSD) 
    save_result_yaml(os.path.join(cfg.result, 'mPSNR_all_checkpoints.yaml'), PSNRM)
    save_result_yaml(os.path.join(cfg.result, 'mSSIM_all_checkpoints.yaml'), SSIMM)
    save_result_yaml(os.path.join(cfg.result, 'mLPIPS_all_checkpoints.yaml'), LPIPSM)            
    swriter.close()


if __name__ == '__main__':
    eval_model(render_folder_name='eval')

# launch tensorboard
# tensorboard --logdir=/home/djchen/PROJECTS/HumanNeRF/experiments/superhumannerf/zju_mocap/p387/single_gpu_baseline_nohash/logs --port 8123