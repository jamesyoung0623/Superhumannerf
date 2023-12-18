import os

import torch
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

from torch.utils.tensorboard import SummaryWriter

from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim


EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height', 'ray_mask', 'framelist', 'coords']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1

    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def _unpack_weights(weights, patch_masks, div_indices):
    N_patch = len(div_indices) - 1

    patch_weights = torch.zeros((patch_masks.shape[0], patch_masks.shape[1], patch_masks.shape[2], 128), dtype=weights.dtype).cuda() # (N_patch, H, W)
    for i in range(N_patch):
        patch_weights[i, patch_masks[i]] = weights[div_indices[i]:div_indices[i+1]]

    return patch_weights


def _unpack_weights_sum(weights_sum, patch_masks, div_indices):
    N_patch = len(div_indices) - 1

    patch_weights_sum = torch.zeros_like(patch_masks, dtype=weights_sum.dtype) # (N_patch, H, W)
    for i in range(N_patch):
        patch_weights_sum[i, patch_masks[i]] = weights_sum[div_indices[i]:div_indices[i+1]]

    return patch_weights_sum


def _unpack_alpha(alpha, patch_masks, div_indices):
    N_patch = patch_masks.size(0)
    patch_size = patch_masks.size(1)

    patch_alpha = torch.zeros((N_patch, patch_size, patch_size, 128), dtype=alpha.dtype).cuda()
    for i in range(N_patch):
        patch_alpha[i, patch_masks[i]] = alpha[div_indices[i]:div_indices[i+1]]

    return patch_alpha


def _unpack_sigma(sigma, patch_masks, div_indices):
    N_patch = patch_masks.size(0)
    patch_size = patch_masks.size(1)

    patch_sigma = torch.zeros((N_patch, patch_size, patch_size, 128), dtype=sigma.dtype).cuda()
    for i in range(N_patch):
        patch_sigma[i, patch_masks[i]] = sigma[div_indices[i]:div_indices[i+1]]

    return patch_sigma


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
        
        self.swriter = SummaryWriter(os.path.join(self.logdir, 'logs'))
        
        self.mse_map_dict = {}
        self.sample_mask_dict = {}
        

    def get_ckpt_path(self, name):
        return os.path.join(self.logdir, f'{name}.tar')

    def ckpt_exists(self, name):
        return os.path.exists(self.get_ckpt_path(name))

    ######################################################
    ## Training 
    
    def get_img_rebuild_loss(self, loss_names, targets, rgb, weights, weights_sum, alpha, sigma, patch_masks):
        alpha_sum = torch.sum(alpha, dim=3)
        losses = {}
        
        if 'lpips' in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), scale_for_lpips(targets.permute(0, 3, 1, 2)))
            losses['lpips'] = torch.mean(lpips_loss)
        
        if 'rgb_loss' in loss_names:
            losses['rgb_loss'] = torch.mean((rgb - targets) ** 2)

        if 'mask_loss' in loss_names:
            losses['mask_loss'] = torch.mean(((torch.flatten(weights_sum) - torch.flatten(torch.ones_like(weights_sum))) ** 2) * torch.flatten(patch_masks.float()))

        if 'opacity' in loss_names:
            epsilon = torch.tensor(1e-1)
            constant = torch.log(epsilon) + torch.log(1+epsilon)

            losses['opacity'] = torch.mean(torch.log(F.relu(torch.flatten(weights)) + epsilon) + torch.log(F.relu(1.0 - torch.flatten(weights)) + epsilon) - constant, dim=-1)
            losses['opacity'] += torch.mean(torch.log(F.relu(torch.flatten(weights_sum)) + epsilon) + torch.log(F.relu(1.0 - torch.flatten(weights_sum)) + epsilon) - constant, dim=-1)
            #losses['opacity'] = torch.mean(torch.log(F.relu(torch.flatten(alpha_sum)) + epsilon) + torch.log(F.relu(1.0 - torch.flatten(alpha_sum)) + epsilon) - constant, dim=-1)

            #losses['opacity'] = torch.mean(-torch.log(torch.exp(-torch.abs(torch.flatten(weights))) + torch.exp(-torch.abs(1.0 - torch.flatten(weights)))))
            #losses['opacity'] += torch.mean(-torch.log(torch.exp(-torch.abs(torch.flatten(weights_sum))) + torch.exp(-torch.abs(1.0 - torch.flatten(weights_sum)))))
            #losses['opacity'] = torch.mean(-torch.log(torch.exp(-torch.abs(torch.flatten(alpha_sum))) + torch.exp(-torch.abs(1.0 - torch.flatten(alpha_sum)))))
            
            
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

        if 'tvloss_sigma' in loss_names:
            losses['tvloss_sigma'] = tvloss(weights_sum[:, None, ...])

        return losses


    def get_loss(self, net_output, patch_masks, bgcolor, targets, div_indices, coords, frame_name, frameWeight=1):
        lossweights = self.cfg.train.lossweights 
        loss_names = list(lossweights.keys())

        rgb = net_output['rgb']
        depth = net_output['depth']  
        weights = net_output['weights']
        weights_sum = net_output['weights_sum']
        alpha = net_output['alpha']  
        sigma = net_output['sigma']  

        unpacked_weights = _unpack_weights(weights, patch_masks, div_indices)
        unpacked_weights_sum = _unpack_weights_sum(weights_sum, patch_masks, div_indices)
        unpacked_alpha = _unpack_alpha(alpha, patch_masks, div_indices)
        unpacked_sigma = _unpack_sigma(sigma, patch_masks, div_indices)
        unpacked_imgs = _unpack_imgs(rgb, patch_masks, bgcolor, targets, div_indices)
        
        mse_patches = np.sum(((targets - unpacked_imgs) ** 2).cpu().detach().numpy(), axis=3)

        for i, coord in enumerate(coords):
            x_min, x_max, y_min, y_max = coord[0].item(), coord[1].item(), coord[2].item(), coord[3].item()
            self.mse_map_dict[frame_name][y_min:y_max, x_min:x_max] = mse_patches[i]
        
        def norm(mse_map):
            mse_map_norm = (255*(mse_map-mse_map.min())/(mse_map.max()-mse_map.min())).astype(np.uint8)
            return mse_map_norm
        
        mse_map_norm = norm(self.mse_map_dict[frame_name])
        losses = self.get_img_rebuild_loss(loss_names, targets, unpacked_imgs, unpacked_weights, unpacked_weights_sum, unpacked_alpha, unpacked_sigma, patch_masks)

        if self.iter < self.cfg.train.opacity_kick_in_iter:
            losses['opacity'] *= 0.0
            
        train_losses = [ weight * losses[k] for k, weight in lossweights.items() ]
        ori_losses = [ loss for _, loss in losses.items() ]
        
        return frameWeight * sum(train_losses), {loss_names[i]: ori_losses[i] for i in range(len(loss_names))}, mse_map_norm

    def train(self, epoch, train_dataloader):
        self.network.train()

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
                
            # if batch['frame_name'] not in self.sample_mask_dict:
            #     self.sample_mask_dict[batch['frame_name']] = 255*(batch['ray_mask'].numpy().reshape(512, 512).astype(int))
            #     self.sample_mask_dict[batch['frame_name']][batch['subject_mask'].numpy()[:, :, 0] > 0.] = 0
            
            # for coord in batch['coords']:
            #     x_min, x_max, y_min, y_max = coord[0].item(), coord[1].item(), coord[2].item(), coord[3].item()
            #     self.sample_mask_dict[batch['frame_name']][y_min:y_max, x_min:x_max] = 128

            # im1 = Image.fromarray(self.sample_mask_dict[batch['frame_name']].astype('uint8'))
            
            train_loss, ori_loss_dict, mse_map_norm = self.get_loss(
                net_output=net_output,
                patch_masks=data['patch_masks'],
                bgcolor=data['bgcolor'] / 255.,
                targets=data['target_patches'],
                div_indices=data['patch_div_indices'], 
                coords=batch['coords'],
                frame_name=batch['frame_name']
            )
            
            # im2 = Image.fromarray(mse_map_norm).convert('L')
            # dst = Image.new('L', (im1.width + im2.width, im1.height))
            # dst.paste(im1, (0, 0))
            # dst.paste(im2, (im1.width, 0))
            # dst.save('ray_sample.jpg')

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

    def progress_end(self):
        self.network.train()
        self.cfg.perturb = 1

    def progress(self):
        self.progress_begin()
        print('Evaluate Progress Images ...')

        images = []
        psnrls = []
        ssimls = []
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

            psnr_tag = compare_psnr(rendered, truth, data_range=1)
            ssim_tag = compare_ssim(rendered, truth, data_range=1, channel_axis=2)
            lpips_tag = torch.mean(self.lpips(scale_for_lpips(torch.from_numpy(rendered).permute(2, 0, 1)).cuda(), scale_for_lpips(torch.from_numpy(truth).permute(2, 0, 1)).cuda()))*1000

            psnrls.append(psnr_tag)               
            ssimls.append(ssim_tag)               
            lpipsls.append(lpips_tag.cpu().detach().numpy())               
            
            truth = to_8b_image(truth)                               
            rendered = to_8b_image(rendered)                         
            
            images.append(np.concatenate([rendered, truth], axis=1))

            # check if we create empty images (only at the begining of training)
            if self.iter <= 5000 and np.allclose(rendered, np.array(self.cfg.bgcolor), atol=5.):
                exit()       
                
        
        tiled_image = tile_images(images)
        psnr_mean = np.mean(psnrls)
        ssim_mean = np.mean(ssimls)
        lpips_mean = np.mean(lpipsls)
        
        Image.fromarray(tiled_image).save(os.path.join(self.logdir, "iter[{:06}]_psnr[{:.2f}]_ssim[{:.4f}]_lpips[{:.2f}].jpg".format(self.iter, psnr_mean, ssim_mean, lpips_mean)))
        self.swriter.add_scalar('PSNR', psnr_mean, self.iter//self.cfg.progress.progress_interval)
        self.swriter.add_scalar('SSIM', ssim_mean, self.iter//self.cfg.progress.progress_interval)
        self.swriter.add_scalar('LPIPS', lpips_mean, self.iter//self.cfg.progress.progress_interval)

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