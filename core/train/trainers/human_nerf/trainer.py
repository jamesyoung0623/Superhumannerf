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
mse_np = lambda x, y : np.mean((x - y) ** 2)                                             
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(10.)                                  

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask', 'framelist', 'coords']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1

    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def _unpack_T_sum(T_sum, patch_masks, div_indices):
    N_patch = len(div_indices) - 1

    patch_T_sum = torch.zeros_like(patch_masks, dtype=T_sum.dtype) # (N_patch, H, W)
    for i in range(N_patch):
        patch_T_sum[i, patch_masks[i]] = T_sum[div_indices[i]:div_indices[i+1]]

    return patch_T_sum


def scale_for_lpips(image_tensor):
    return image_tensor * 2. - 1.


class Trainer(object):
    def __init__(self, cfg, network, optimizer):
        print('\n********** Init Trainer ***********')
        self.cfg = cfg
        self.logdir = os.path.join('../experiments', cfg.category, cfg.task, cfg.subject, cfg.experiment)
        os.makedirs(self.logdir, exist_ok=True)

        self.network = network.cuda()

        self.optimizer = optimizer
        self.update_lr = create_lr_updater(self.cfg)

        if self.cfg.resume and self.ckpt_exists(self.cfg.load_net):
            self.load_ckpt(f'{self.cfg.load_net}')
        else:
            self.iter = 0
            self.save_ckpt('init')
            self.iter = 1

        self.timer = Timer()

        if "lpips" in self.cfg.train.lossweights.keys():
            self.lpips = LPIPS(net='vgg')
            set_requires_grad(self.lpips, requires_grad=True)
            self.lpips = self.lpips.cuda()

        print("Load Progress Dataset ...")
        self.prog_dataloader = create_dataloader(self.cfg, 'progress')

        print('************************************')
        
        #self.test_loader = create_dataloader(self.cfg, 'movement')
        self.swriter = SummaryWriter(os.path.join(self.logdir, 'logs'))
        
        self.mse_map_dict = {}
        self.sample_mask_dict = {}
        

    def get_ckpt_path(self, name):
        return os.path.join(self.logdir, f'{name}.tar')

    def ckpt_exists(self, name):
        return os.path.exists(self.get_ckpt_path(name))

    ######################################################
    ## Training 
    
    def get_img_rebuild_loss(self, loss_names, targets, rgb, T_sum, sigma, patch_masks):
        losses = {}
        if 'lpips' in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), scale_for_lpips(targets.permute(0, 3, 1, 2)))
            losses['lpips'] = torch.mean(lpips_loss)
        
        if 'mse' in loss_names:
            losses['mse'] = img2mse(rgb, targets)

        if 'opacity' in loss_names:
            epsilon = torch.tensor(1e-1)
            constant = torch.log(epsilon) + torch.log(1+epsilon)

            losses['opacity'] = torch.mean(torch.log(F.relu(torch.flatten(T_sum)) + epsilon) + torch.log(F.relu(1.0 - torch.flatten(T_sum)) + epsilon) - constant, dim=-1)
            # lossBCE = -torch.mean( torch.mul( F.relu(torch.flatten(alpha)), torch.log(F.relu(torch.flatten(alpha))) ) + torch.mul( F.relu(1.-torch.flatten(alpha)), torch.log(F.relu(1.-torch.flatten(alpha))) ), dim=-1 )
            
            # loss = nn.BCELoss()
            # lossBCE = loss( torch.flatten(alpha), torch.flatten(patch_masks.float()) )


        if 'l1' in loss_names:
            losses['l1'] = img2l1(rgb, targets)

        def tvloss(x):
            batch_size = x.size()[0]
            h_x = x.size()[2]
            w_x = x.size()[3]
            count_h = x[:,:,1:,:].size()[1]*x[:,:,1:,:].size()[2]*x[:,:,1:,:].size()[3]
            count_w = x[:,:,:,1:].size()[1]*x[:,:,:,1:].size()[2]*x[:,:,:,1:].size()[3]
            h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]), 2).sum()
            w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]), 2).sum()

            return 2*(h_tv/count_h+w_tv/count_w)/batch_size

        if 'tvloss_rgb' in loss_names:
            losses['tvloss_rgb'] = tvloss(rgb)

        #if 'tvloss_sigma' in loss_names:
        #    print(sigma[:, None, ...].size())
        #    losses['tvloss_sigma'] = tvloss(sigma)
        #    print(losses['tvloss_sigma'])
        #    exit()

        return losses


    def get_loss(self, net_output, patch_masks, bgcolor, targets, div_indices, coords, frame_name, frameWeight=1):
        lossweights = self.cfg.train.lossweights 
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        depth = net_output['depth']  
        T = net_output['T']
        T_sum = net_output['T_sum']
        alpha = net_output['alpha']  
        sigma = net_output['sigma']  

        unpacked_T_sum = _unpack_T_sum(T_sum, patch_masks, div_indices)
        unpacked_imgs = _unpack_imgs(rgb, patch_masks, bgcolor, targets, div_indices)
        
        mse_patches = np.sum(((targets - unpacked_imgs) ** 2).cpu().detach().numpy(), axis=3)

        for i, coord in enumerate(coords):
            x_min, x_max, y_min, y_max = coord[0].item(), coord[1].item(), coord[2].item(), coord[3].item()
            self.mse_map_dict[frame_name][y_min:y_max, x_min:x_max] = mse_patches[i]
        
        def norm(mse_map):
            mse_map_norm = (255*(mse_map-mse_map.min())/(mse_map.max()-mse_map.min())).astype(np.uint8)
            return mse_map_norm
        
        mse_map_norm = norm(self.mse_map_dict[frame_name])

        losses = self.get_img_rebuild_loss(loss_names, targets, unpacked_imgs, unpacked_T_sum, sigma, patch_masks)

        if self.iter < self.cfg.train.opacity_kick_in_iter:
            losses['opacity'] *= 0.0
            
        train_losses = [ weight * losses[k] for k, weight in lossweights.items() ]
        ori_losses = [ loss for _, loss in losses.items() ]
        
        return frameWeight * sum(train_losses), {loss_names[i]: ori_losses[i] for i in range(len(loss_names))}, mse_map_norm

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1
        self.network.train()
        self.training = True

    def train_end(self):
        pass

    def train(self, epoch, train_dataloader):
        self.train_begin(train_dataloader=train_dataloader)

        self.timer.begin()
        for batch_idx, batch in enumerate(train_dataloader):
            if self.iter > self.cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            net_output, distloss = self.network(self.iter, **data)
            
            if batch['frame_name'] not in self.mse_map_dict:
                self.mse_map_dict[batch['frame_name']] = np.zeros((512, 512))
                
            if batch['frame_name'] not in self.sample_mask_dict:
                self.sample_mask_dict[batch['frame_name']] = 255*(batch['ray_mask'].numpy().reshape(512, 512).astype(int))
                self.sample_mask_dict[batch['frame_name']][batch['subject_mask'].numpy()[:, :, 0] > 0.] = 0
            
            for coord in batch['coords']:
                x_min, x_max, y_min, y_max = coord[0].item(), coord[1].item(), coord[2].item(), coord[3].item()
                self.sample_mask_dict[batch['frame_name']][y_min:y_max, x_min:x_max] = 128

            im1 = Image.fromarray(self.sample_mask_dict[batch['frame_name']].astype('uint8'))
            
            train_loss, ori_loss_dict, mse_map_norm = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'], 
                coords=batch['coords'],
                frame_name=batch['frame_name']
            )
            
            im2 = Image.fromarray(mse_map_norm).convert('L')
            dst = Image.new('L', (im1.width + im2.width, im1.height))
            dst.paste(im1, (0, 0))
            dst.paste(im2, (im1.width, 0))
            dst.save('ray_sample.jpg')
            #im2.save('./mse_map/{}.jpg'.format(batch['frame_name']))

            # ori_loss_dict['distloss'] = distloss
            # train_loss += distloss
            
            train_loss.backward()
            self.optimizer.step()

            if self.iter % self.cfg.train.log_interval == 0:
                loss_str = f"Loss: {train_loss.item():.4f} [ "
                for k, v in ori_loss_dict.items():
                    loss_str += f"{k}:{v.item():>2.4f} "
                loss_str += "]"

                log_str = 'Epoch: {:>3d} [Iter {:>5d}, {:>3d}/{:03d} ({:>3.0f}%), {}] {}'
                log_str = log_str.format(
                    epoch, self.iter,
                    (batch_idx+1) * self.cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str
                )
                print(log_str)

            if self.iter % self.cfg.progress.progress_interval == 0:
                self.progress()
                self.save_ckpt(f'iter_{self.iter}')
                    
            self.update_lr(self.cfg, self.optimizer, self.iter)
            self.iter += 1
        
        self.swriter.close()


    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        self.cfg.perturb = 0.
        self.training = False

    def progress_end(self):
        self.network.train()
        self.cfg.perturb = 1

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

            rendered = np.full((height * width, 3), np.array(self.cfg.bgcolor)/255., dtype='float32')
            truth = np.full((height * width, 3), np.array(self.cfg.bgcolor)/255., dtype='float32')
            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU + ['target_rgbs'])
            with torch.no_grad():
                net_output, distloss = self.network(self.iter, **data)
                
            rgb = net_output['rgb'].data.to("cpu").numpy()
            target_rgbs = batch['target_rgbs']

            rendered[ray_mask] = rgb
            truth[ray_mask] = target_rgbs
            
            truth = truth.reshape((height, width, -1))               
            rendered = rendered.reshape((height, width, -1))   

            mse = mse_np(rendered, truth)

            psnr_tag = mse2psnr_np(mse)
            lpips_tag = torch.mean(self.lpips(scale_for_lpips(torch.from_numpy(rendered).permute(2, 0, 1)).cuda(), scale_for_lpips(torch.from_numpy(truth).permute(2, 0, 1)).cuda()))*1000

            psnrls.append(psnr_tag)               
            lpipsls.append(lpips_tag.cpu().detach().numpy())               
            
            truth = to_8b_image(truth)                               
            rendered = to_8b_image(rendered)                         
            
            images.append(np.concatenate([rendered, truth], axis=1))

            # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                exit()       
                
        
        tiled_image = tile_images(images)
        psnr_mean = np.mean(psnrls)
        lpips_mean = np.mean(lpipsls)
        
        Image.fromarray(tiled_image).save(os.path.join(self.logdir, "iter[{:06}]_psnr[{:.2f}]_lpips[{:.2f}].jpg".format(self.iter, psnr_mean, lpips_mean)))
        self.swriter.add_scalar('PSNR', psnr_mean, self.iter//self.cfg.progress.progress_interval)
        self.swriter.add_scalar('LPIPS', lpips_mean, self.iter//self.cfg.progress.progress_interval)

        #if self.iter % self.cfg.progress.eval_interval == 0:
        #    self.eval_model(render_folder_name='eval', n=self.iter/self.cfg.progress.eval_interval)
        
        self.progress_end()
        return


    ######################################################3
    ## Utils

    def save_ckpt(self, name):
        path = self.get_ckpt_path(name)
        print(f"Save checkpoint to {path} ...")

        torch.save({'iter': self.iter, 'network': self.network.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load_ckpt(self, name):
        path = self.get_ckpt_path(name)
        print(f"Load checkpoint from {path} ...")
        
        ckpt = torch.load(path, map_location='cuda:0')
        self.iter = ckpt['iter'] + 1

        self.network.load_state_dict(ckpt['network'], strict=False)
        self.optimizer.load_state_dict(ckpt['optimizer'])

    ######################################################3
    ## Eval
    
    def load_network(self, checkpoint):
        model = create_network()
        ckpt_path = os.path.join(self.logdir, f'{checkpoint}.tar')
        ckpt = torch.load(ckpt_path, map_location='cuda:0')
        model.load_state_dict(ckpt['network'], strict=False)
        print('load network from ', ckpt_path)
        return model.cuda()

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
        processed_pred = torch.from_numpy(pred).float().unsqueeze(0).cuda() * 2. - 1.
        processed_target = torch.from_numpy(target).float().unsqueeze(0).cuda() * 2. - 1.

        lpips_loss = model(processed_pred.permute(0, 3, 1, 2), processed_target.permute(0, 3, 1, 2))
        return torch.mean(lpips_loss).cpu().detach().item()

    def eval_model(self, render_folder_name='eval', n=0):
        set_requires_grad(self.lpips, requires_grad=False)

        out_folder_name = f'iter_{self.iter}_{render_folder_name}'
        
        self.writer = ImageWriter(output_dir=self.logdir, exp_name=out_folder_name)

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
                net_output, distloss = self.network(self.iter, **data)
            rgb = net_output['rgb']
            alpha = net_output['alpha']

            rgb_img, alpha_img, truth_img = self.unpack_to_image(
                width, height, ray_mask, np.array(self.cfg.bgcolor) / 255.,
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
        
