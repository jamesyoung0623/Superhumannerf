import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm

from third_parties.lpips import LPIPS

from core.train import create_lr_updater
from core.data import create_dataloader
from core.utils.network_util import set_requires_grad
from core.utils.train_util import cpu_data_to_gpu, Timer
from core.utils.image_util import tile_images, to_8b_image

from configs import cfg
from datetime import datetime

import skimage

from core.nets import create_network
from core.utils.image_util import ImageWriter, to_8b_image, to_8b3ch_image

from torch.utils.tensorboard import SummaryWriter

import glob
import yaml

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device)) 
# to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)
mse_np = lambda x, y : np.mean((x - y) ** 2)                                             
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(10.)                                  

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask', 'framelist']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1

    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def _unpack_alpha(alphas, patch_masks, div_indices):
    N_patch = len(div_indices) - 1

    patch_alphas = torch.zeros_like(patch_masks, dtype=alphas.dtype) # (N_patch, H, W)
    for i in range(N_patch):
        patch_alphas[i, patch_masks[i]] = alphas[div_indices[i]:div_indices[i+1]]

    return patch_alphas


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, network, optimizer):
        print('\n********** Init Trainer ***********')
        network = network.cuda().deploy_mlps_to_secondary_gpus()
        self.network = network

        self.optimizer = optimizer
        self.update_lr = create_lr_updater()

        if cfg.resume and Trainer.ckpt_exists(cfg.load_net):
            self.load_ckpt(f'{cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        self.timer = Timer()

        if "lpips" in cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=True)
            self.lpips = nn.DataParallel(self.lpips).cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(data_type='progress')

        print('************************************')
        
        self.test_loader = create_dataloader('movement')
        log_dir = os.path.join(cfg.logdir, 'logs')
        self.swriter = SummaryWriter(log_dir)


    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################
    ## Training 
    
    def get_img_rebuild_loss(self, loss_names, target, patch_masks, alpha, rgb):
        losses = {}
        # rgb/target: [nPatch, patchSize, patchSize, RGB]
        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)
        
        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "opacity" in loss_names:
            #epsilon = torch.tensor(1e-3)
            #epsilon = torch.tensor(1e-2) # weight 0.0001
            epsilon = torch.tensor(1e-1) # weight 0.005
            #epsilon = torch.tensor(0.58) # weight 0.05
            #epsilon = torch.tensor(1.0)  # weight 0.1

            constant = torch.log(epsilon) + torch.log(1+epsilon)
            # rayalpha : NCHW see Neural Volumes (ACM TOG 2019)/ personNeRF (CVPR 2023)
            # dj: use relu to ignore negative values !!! (prevent NaN)
            # losses["opacity"] = torch.mean( torch.log(alpha+epsilon) + torch.log(F.relu(1.-alpha)+epsilon) - constant, dim=-1)

            loss2 = torch.mean( torch.log(F.relu(torch.flatten(alpha)) + epsilon) + torch.log(F.relu(1. - torch.flatten(alpha)) + epsilon) - constant, dim=-1)
            #loss2 = torch.mean( torch.log(torch.flatten(alpha) + epsilon) + torch.log(1. - torch.flatten(alpha) + epsilon) - constant, dim=-1)

            # lossBCE = -torch.mean( torch.mul( F.relu(torch.flatten(alpha)), torch.log(F.relu(torch.flatten(alpha))) ) + torch.mul( F.relu(1.-torch.flatten(alpha)), torch.log(F.relu(1.-torch.flatten(alpha))) ), dim=-1 )
            
            # loss = nn.BCELoss()
            # lossBCE = loss( torch.flatten(alpha), torch.flatten(patch_masks.float()) )
            # print(lossBCE)
            # print(alpha[patch_masks].shape)

            # loss3 = img2mse(alpha,patch_masks.float()) # bad results
            # loss3 = img2l1(alpha,patch_masks.float()) # weight 0.2

            losses["opacity"] = loss2
            # if np.isnan(losses["opacity"].cpu().detach().numpy()):
            #     F.relu(torch.flatten(alpha))
            #     1. - F.relu(torch.flatten(alpha))
            #     torch.log(alpha)
            #     torch.log(1-alpha)
            #     torch.isnan( torch.log(F.relu(torch.flatten(alpha))) ).int().sum()
            #     torch.isnan( torch.log(epsilon + 1. - F.relu(torch.flatten(alpha)))[1000:1200] ).int().sum()
            #     torch.isnan(losses["opacity"]).int().sum()
            #     cumLoss = torch.mean( torch.log(F.relu(alpha[1])+epsilon) + torch.log(F.relu(1.-alpha[1])+epsilon) - constant)
            #     loss2 = torch.mean( torch.log(0.1 + F.relu(torch.flatten(alpha))) + torch.log(0.1 + 1. - F.relu(torch.flatten(alpha))), dim=-1)
            #     torch.mean( torch.log(torch.tensor(0.1)) + torch.log(torch.tensor(0.1 + 1.)), dim=-1)
            #     torch.mean( torch.log( F.relu(torch.flatten(alpha))) + torch.log(1. - F.relu(torch.flatten(alpha))), dim=-1)


        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "psnr" in loss_names:
            losses["psnr"] = mse2psnr(img2mse(rgb, target))[0]
        
        return losses

    def get_loss(self, net_output, patch_masks, bgcolor, targets, div_indices, frameWeight=1):
        lossweights = cfg.train.lossweights 
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']     # [2560, 3] = (N_patch, H, W, 3)
        alpha = net_output['alpha'] # [2560] = (N_patch, H, W, 1)
        depth = net_output['depth'] # [2560] = (N_patch, H, W, 1)

        losses = self.get_img_rebuild_loss(loss_names, targets, patch_masks, _unpack_alpha(alpha, patch_masks, div_indices), _unpack_imgs(rgb, patch_masks, bgcolor, targets, div_indices))

        if self.iter < cfg.train.opacity_kick_in_iter:
            losses['opacity'] *= 0.0
        
        train_losses = [ weight * losses[k] for k, weight in lossweights.items() ]
        ori_losses = [ loss for _, loss in losses.items() ]
        
        return frameWeight * sum(train_losses), {loss_names[i]: ori_losses[i] for i in range(len(loss_names))}

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1
        self.network.train()

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        #TakeDistinctPoses = cfg.motionCLIP.training_frames_poseDistinct
        #LossDistinctness = cfg.motionCLIP.training_frames_lossDistinct
        self.train_begin(train_dataloader=train_dataloader)

        #if TakeDistinctPoses or LossDistinctness:
        # skip normal-motion frames ###################################
        #    from sklearn.neighbors import LocalOutlierFactor
        #    training_percent = 0.5
        #    lof = LocalOutlierFactor(n_neighbors=20, algorithm='auto', metric='minkowski', metric_params=None, contamination=training_percent, novelty=False, n_jobs=-1)
        #    for batch_idx, batch in enumerate(train_dataloader): # only run one iteration
                
        #        motionCLIP = batch['motionCLIP']
        #        framelist = batch['framelist']
                
        #        outliers = lof.fit_predict(torch.squeeze(motionCLIP))
        #        outlier_index = np.where(outliers==-1)

        #        candidates = []
        #        keepIdx = outlier_index[0].tolist()
        #        outliers_value = lof.negative_outlier_factor_ 
        #        frame_weight = - outliers_value

        #        for i in range(len(keepIdx)):
        #            candidates.append(framelist[keepIdx[i]][0])
                    
        #        break    
        #    print('keep %f data' % training_percent)
        ###############################################################
        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):
        #    if TakeDistinctPoses:
        #        if batch['frame_name'][0] not in candidates: 
        #            continue                                 

            if self.iter > cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]
            # ['frame_name', 'img_width', 'img_height', 'ray_mask', 'rays', 'near', 'far', 'bgcolor', 'patch_div_indices', 'patch_masks', 'target_patches', 'dst_Rs', 'dst_Ts', 'cnl_gtfms', 'motion_weights_priors', 'cnl_bbox_min_xyz', 'cnl_bbox_max_xyz', 'cnl_bbox_scale_xyz', 'dst_posevec', 'motionCLIP', 'framelist']   
            

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            
            net_output = self.network(**data)

            frameWeightValue = 1.0
            #if LossDistinctness: 
            #    frameWeightValue = frame_weight[batch_idx]
            
            train_loss, ori_loss_dict = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'], 
                frameWeight=frameWeightValue
            )

            train_loss.backward()
            self.optimizer.step()

            if self.iter % cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} [ "
                for k, v in ori_loss_dict.items():
                    loss_str += f"{k}:{v.item():>2.4f} "
                loss_str += "]"

                log_str = 'Epoch: {:>3d} [Iter {:>5d}, {:>3d}/{:03d} ({:>3.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    (batch_idx+1) * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str
                )
                print(log_str)

            if self.iter % cfg.progress.dump_interval == 0:
                self.progress()
                self.save_ckpt(f'iter_{self.iter}')
                    
            self.update_lr(self.optimizer, self.iter)
            self.iter += 1
        
        self.swriter.close()
        

    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = 1

    def progress(self):
        self.progress_begin()
        print('Evaluate Progress Images ...')

        images = []
        psnrls = []
        lpipsls = []

        for _, batch in enumerate(tqdm(self.prog_dataloader)):
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            # cfg.bgcolor = [100, 100, 250] 
            rendered = np.full((height * width, 3), np.array(cfg.bgcolor)/255., dtype='float32')
            truth = np.full((height * width, 3), np.array(cfg.bgcolor)/255., dtype='float32')
            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data)

            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs

            truth = truth.reshape((height, width, -1))               
            rendered = rendered.reshape((height, width, -1))   

            mse = mse_np(rendered, truth)
            psnr_tag = mse2psnr_np(mse)
            lpips_tag = torch.mean(self.lpips(scale_for_lpips(torch.from_numpy(rendered).permute(2, 0, 1)), scale_for_lpips(torch.from_numpy(truth).permute(2, 0, 1))))*1000

            psnrls.append(psnr_tag)               
            lpipsls.append(lpips_tag.cpu().detach().numpy())               
            
            truth = to_8b_image(truth)                               
            rendered = to_8b_image(rendered)                         
            
            images.append(np.concatenate([rendered, truth], axis=1))

            # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and np.allclose(rendered, np.array(cfg.bgcolor), atol=5.):
                exit()       
                
        
        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(os.path.join(cfg.logdir, "iter[{:06}]_psnr[{:.2f}]_lpips[{:.2f}].jpg".format(self.iter, np.mean(psnrls), np.mean(lpipsls))))

        self.eval_model(render_folder_name='eval', n=self.iter/cfg.progress.dump_interval-1)
        self.progress_end()
        return


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({
            'iter': self.iter,
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load_ckpt(self, name):
        path = Trainer.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])

    ######################################################3
    ## Eval
    
    def load_network(self, checkpoint):
        model = create_network()
        ckpt_path = os.path.join(cfg.logdir, f'{checkpoint}.tar')
        ckpt = torch.load(ckpt_path, map_location='cuda:0')
        model.load_state_dict(ckpt['network'], strict=False)
        print('load network from ', ckpt_path)
        return model.cuda().deploy_mlps_to_secondary_gpus()

    def unpack_alpha_map(self, alpha_vals, ray_mask, width, height):
        alpha_map = np.zeros((height * width), dtype='float32')
        alpha_map[ray_mask] = alpha_vals
        return alpha_map.reshape((height, width))

    def unpack_to_image(self, width, height, ray_mask, bgcolor, rgb, alpha, truth=None):
        rgb_image = np.full((height * width, 3), bgcolor, dtype='float32')
        truth_image = np.full((height * width, 3), bgcolor, dtype='float32')

        rgb_image[ray_mask] = rgb
        rgb_image = to_8b_image(rgb_image.reshape((height, width, 3)))

        if truth is not None:
            truth_image[ray_mask] = truth
            truth_image = to_8b_image(truth_image.reshape((height, width, 3)))

        alpha_map = self.unpack_alpha_map(alpha, ray_mask, width, height)
        alpha_image = to_8b3ch_image(alpha_map)

        return rgb_image, alpha_image, truth_image

    def psnr_metric(self, img_pred, img_gt):
        ''' Calculate psnr metric
            Args:
                img_pred: ndarray, W*H*3, range 0-1
                img_gt: ndarray, W*H*3, range 0-1

            Returns:
                psnr metric: scalar
        '''
        mse = np.mean((img_pred - img_gt) ** 2)
        psnr = -10 * np.log(mse) / np.log(10)
        return psnr.item()

    def lpips_metric(self, model, pred, target):
        # convert range from 0-1 to -1-1
        processed_pred = torch.from_numpy(pred).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.
        processed_target = torch.from_numpy(target).float().unsqueeze(0).to(cfg.primary_gpus[0]) * 2. - 1.

        lpips_loss = model(processed_pred.permute(0, 3, 1, 2), processed_target.permute(0, 3, 1, 2))
        return torch.mean(lpips_loss).cpu().detach().item()

    def eval_model(self, render_folder_name='eval', n=0):
        set_requires_grad(self.lpips, requires_grad=False)

        out_folder_name = f'iter_{self.iter}_{render_folder_name}'
        #if os.path.isdir(os.path.join(cfg.logdir, out_folder_name)):
        #    continue
        
        self.writer = ImageWriter(output_dir=cfg.logdir, exp_name=out_folder_name)

        PSNRA = []
        SSIMA = []
        LPIPSA = []
        for idx, batch in enumerate(self.test_loader):
            for k, v in batch.items():
                batch[k] = v[0]
            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            # prediction
            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output = self.network(**data, iter_val=cfg.eval_iter)
            rgb = net_output['rgb']
            alpha = net_output['alpha']

            rgb_img, alpha_img, truth_img = self.unpack_to_image(
                width, height, ray_mask, np.array(cfg.bgcolor) / 255.,
                rgb.data.cpu().numpy(),
                alpha.data.cpu().numpy(),
                batch['target_rgbs']
            )
                
            imgs = [rgb_img, truth_img, alpha_img]
            img_out = np.concatenate(imgs, axis=1)
            self.writer.append(img_out, img_name=batch['frame_name'])
            rgb_img_norm = rgb_img / 255.
            truth_img_norm = truth_img / 255.

            psnr = self.psnr_metric(rgb_img_norm, truth_img_norm)
            ssim = skimage.metrics.structural_similarity(rgb_img_norm, truth_img_norm, data_range=1, channel_axis=2)
            lpips = self.lpips_metric(model=self.lpips, pred=rgb_img_norm, target=truth_img_norm)
            print(f"Checkpoint [iter_{self.iter}] - {idx}/{len(self.test_loader)}: PSNR is {psnr:.4f}, SSIM is {ssim:.4f}, LPIPS is {lpips*1000:.4f}")

            PSNRA.append(psnr)
            SSIMA.append(ssim)
            LPIPSA.append(lpips)
        
        psnr_mean = np.mean(PSNRA).item()
        ssim_mean = np.mean(SSIMA).item()
        lpips_mean = np.mean(LPIPSA).item()
        print(f"Checkpoint [iter_{self.iter}]: mPSNR is {psnr_mean}, mSSIM is {ssim_mean}, mLPIPS is {lpips_mean*1000}")
        self.writer.finalize()

        self.swriter.add_scalar('AllCP_PSNR', psnr_mean, n)
        self.swriter.add_scalar('AllCP_SSIM', ssim_mean, n)
        self.swriter.add_scalar('AllCP_LPIPS', lpips_mean*1000, n)

