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

img2mse = lambda x, y : torch.mean((x - y) ** 2)
img2l1 = lambda x, y : torch.mean(torch.abs(x-y))
mse2psnr = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).to(x.device)) # dj
# to8b = lambda x : (255.*np.clip(x,0.,1.)).astype(np.uint8)
mse_np = lambda x, y : np.mean((x - y) ** 2)                                             # dj
mse2psnr_np = lambda x : -10. * np.log(x) / np.log(10.)                                  # dj

EXCLUDE_KEYS_TO_GPU = ['frame_name', 'img_width', 'img_height']


def _unpack_imgs(rgbs, patch_masks, bgcolor, targets, div_indices):
    N_patch = len(div_indices) - 1

    assert patch_masks.shape[0] == N_patch
    assert targets.shape[0] == N_patch

    patch_imgs = bgcolor.expand(targets.shape).clone() # (N_patch, H, W, 3)
    for i in range(N_patch):
        patch_imgs[i, patch_masks[i]] = rgbs[div_indices[i]:div_indices[i+1]]

    return patch_imgs


def _unpack_alpha(alphas, patch_masks, div_indices): # dj
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

    @staticmethod
    def get_ckpt_path(name):
        return os.path.join(cfg.logdir, f'{name}.tar')

    @staticmethod
    def ckpt_exists(name):
        return os.path.exists(Trainer.get_ckpt_path(name))

    ######################################################3
    ## Training 
    def get_img_rebuild_loss(self, loss_names, target, patch_masks, alpha, rgb):
        losses = {}
        # breakpoint()
        # dj: rgb/target: [nPatch, patchSize, patchSize, RGB]
        if "lpips" in loss_names:
            lpips_loss = self.lpips(scale_for_lpips(rgb.permute(0, 3, 1, 2)), scale_for_lpips(target.permute(0, 3, 1, 2)))
            losses["lpips"] = torch.mean(lpips_loss)
        
        if "mse" in loss_names:
            losses["mse"] = img2mse(rgb, target)

        if "opacity" in loss_names:
            # breakpoint()
            N_patch = alpha.shape[0] - 1
            epsilon1 = torch.tensor(1e-7)
            epsilon2 = torch.tensor(1e-2) # weight 0.0001
            epsilon3 = torch.tensor(1e-1) # weight 0.005
            epsilon4 = torch.tensor(0.58) # weight 0.05
            epsilon0 = torch.tensor(1.0)  # weight 0.1
            epsilon  = epsilon3
            constant = torch.log(epsilon) + torch.log(1+epsilon)
            # rayalpha : NCHW see Neural Volumes (ACM TOG 2019)/ personNeRF (CVPR 2023)
            # dj: use relu to ignore negative values !!! (prevent NaN)
            # losses["opacity"] = torch.mean( torch.log(alpha+epsilon) + torch.log(F.relu(1.-alpha)+epsilon) - constant, dim=-1)

            # loss1 = torch.sum( torch.log(F.relu(torch.flatten(alpha))+epsilon) + torch.log(F.relu(1.-torch.flatten(alpha))+epsilon) ) - constant
            # print(loss1)
            # cumLoss = torch.tensor(0.).cuda()
            # for i in range(N_patch):
            #     curLoss = torch.sum( torch.log(F.relu(alpha[i])+epsilon) + torch.log(F.relu(1.-alpha[i])+epsilon) ) - constant
            #     # print(curLoss)
            #     cumLoss += curLoss
            # print(cumLoss)
            loss2 = torch.mean( torch.log(epsilon + F.relu(torch.flatten(alpha))) + torch.log(epsilon + F.relu(1. - torch.flatten(alpha))) - constant, dim=-1)
            # print(loss2)
            # lossBCE = -torch.mean( torch.mul( F.relu(torch.flatten(alpha)), torch.log(F.relu(torch.flatten(alpha))) ) + torch.mul( F.relu(1.-torch.flatten(alpha)), torch.log(F.relu(1.-torch.flatten(alpha))) ), dim=-1 )
            
            # loss = nn.BCELoss()
            # lossBCE = loss( torch.flatten(alpha), torch.flatten(patch_masks.float()) )
            # print(lossBCE)
            # print(alpha[patch_masks].shape)

            # loss3 = img2mse(alpha,patch_masks.float()) # bad results
            # loss3 = img2l1(alpha,patch_masks.float()) # weight 0.2

            # breakpoint()
            losses["opacity"] = loss2
            # if np.isnan(losses["opacity"].cpu().detach().numpy()):
            #     breakpoint()
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

            # breakpoint()


        if "l1" in loss_names:
            losses["l1"] = img2l1(rgb, target)

        if "psnr" in loss_names:  # dj
            losses["psnr"] = mse2psnr(img2mse(rgb, target))[0] # dj
        return losses

    def get_loss(self, net_output, patch_masks, bgcolor, targets, div_indices, frameWeight=1):

        lossweights = cfg.train.lossweights
        loss_names = list(lossweights.keys())
        # print(loss_names)
        try: # dj
            rgb = net_output['rgb']     # [2560,3] = (N_patch, H, W, 3)
            alpha = net_output['alpha'] # [2560] = (N_patch, H, W, 1)
            depth = net_output['depth'] # [2560] = (N_patch, H, W, 1)
        except:
            breakpoint()
        # print(alpha)
        # breakpoint()
        losses = self.get_img_rebuild_loss(loss_names, targets, patch_masks, _unpack_alpha(alpha, patch_masks, div_indices), _unpack_imgs(rgb, patch_masks, bgcolor, targets, div_indices))
        # print(losses)
        # breakpoint()  
        train_losses = [ weight * losses[k] for k, weight in lossweights.items() ]

        ori_losses = [ losses[k] for k, weight in lossweights.items() ] # dj
        # losses['opacity'] * 10
        # train_losses = train_losses + losses['opacity'] * 10
        # print(train_losses)
        # print(ori_losses)
        # breakpoint()
        return frameWeight * sum( train_losses ), {loss_names[i]: train_losses[i] for i in range(len(loss_names))}, {loss_names[i]: ori_losses[i] for i in range(len(loss_names))} # dj

    def train_begin(self, train_dataloader):
        assert train_dataloader.batch_size == 1

        self.network.train()
        cfg.perturb = cfg.train.perturb

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
        #        if batch['frame_name'][0] not in candidates:  # dj
        #            continue                                  # dj

            if self.iter > cfg.train.maxiter:
                break

            self.optimizer.zero_grad()

            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            batch['iter_val'] = torch.full((1,), self.iter)
            data = cpu_data_to_gpu(batch, exclude_keys=EXCLUDE_KEYS_TO_GPU)
            
            net_output = self.network(**data)
            if net_output == None:
                continue

            frameWeightValue = 1.0
            #if LossDistinctness: 
            #    frameWeightValue = frame_weight[batch_idx]
            
            train_loss, loss_dict, ori_loss_dict = self.get_loss(
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

                log_str = 'Epoch: {:>3d} [Iter {:>5d}, {:>3d}/{:03d} ({:>3.0f}%), {}] {}' # dj
                log_str = log_str.format(
                    epoch, self.iter,
                    (batch_idx+1) * cfg.train.batch_size, len(train_dataloader.dataset),
                    100. * batch_idx / len(train_dataloader), 
                    self.timer.log(),
                    loss_str)
                print(log_str)

            is_reload_model = False

            if self.iter in [200, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 7000, 8000, 9000, 10000] or self.iter % cfg.progress.dump_interval == 0:
                is_reload_model = self.progress()

            if not is_reload_model:
                if self.iter % cfg.train.save_checkpt_interval == 0:
                    self.save_ckpt('latest')

                if cfg.save_all:
                    if self.iter % cfg.train.save_model_interval == 0:
                        self.save_ckpt(f'iter_{self.iter}')

                self.update_lr(self.optimizer, self.iter)

                self.iter += 1

    def finalize(self):
        self.save_ckpt('latest')

    ######################################################3
    ## Progress

    def progress_begin(self):
        self.network.eval()
        cfg.perturb = 0.

    def progress_end(self):
        self.network.train()
        cfg.perturb = cfg.train.perturb

    def progress(self):
        self.progress_begin()

        print('Evaluate Progress Images ...')

        images = []
        psnrls = []
        lpipsls = []
        is_empty_img = False
        for _, batch in enumerate(tqdm(self.prog_dataloader)):
            # only access the first batch as we process one image one time
            for k, v in batch.items():
                batch[k] = v[0]

            width = batch['img_width']
            height = batch['img_height']
            ray_mask = batch['ray_mask']

            # cfg.bgcolor = [100, 100, 250] # dj
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
                is_empty_img = True
                break       
                
        
        tiled_image = tile_images(images)
        
        Image.fromarray(tiled_image).save(os.path.join(cfg.logdir, "iter[{:06}]_psnr[{:.2f}]_lpips[{:.2f}].jpg".format(self.iter, np.mean(psnrls), np.mean(lpipsls))))

        if is_empty_img:
            print("Produce empty images; reload the init model.")
            self.load_ckpt('init')
            
        self.progress_end()

        return is_empty_img


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
