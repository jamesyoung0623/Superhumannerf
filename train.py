from configs import cfg

from core.utils.log_util import Logger
from core.data import create_dataloader
from core.nets import create_network
from core.train import create_trainer, create_optimizer

from datetime import datetime

import os
import random
import numpy as np
import torch
import torch.cuda
import torch.backends.cudnn


def make_deterministic(seed):
    '''Seed everything for better reproducibility.
    (some pytorch operation is non-deterministic like the backprop of grid_samples)
    '''
    seed=int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # sets the seed for generating random numbers.
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True,warn_only=True)
    # torch.autograd.set_detect_anomaly(True) # for debug, slowing the process

    os.environ["PYTHONHASHSEED"] = str(seed) # for hash function
    os.environ["CUDA_LAUNCH_BLOCKING"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # modify the function _worker_init_fn defined in /home/djchen/PROJECTS/HumanNeRF/superhumannerf/core/data/create_dataset.py
    # see https://pytorch.org/docs/stable/notes/randomness.html#pytorch
    
def main():
    log = Logger()
    log.print_config()
    print('fixed random seed: %d' % cfg.train.seed)
    make_deterministic(cfg.train.seed)
    model = create_network()
    optimizer = create_optimizer(model)
    trainer = create_trainer(model, optimizer)
    train_loader = create_dataloader('train')

    # estimate start epoch
    tic = datetime.now()
    epoch = trainer.iter // len(train_loader) + 1
    while not trainer.iter > cfg.train.maxiter:
        trainer.train(epoch=epoch, train_dataloader=train_loader)
        epoch += 1

    trainer.finalize()
    toc = datetime.now()
    print('# # '*20)
    print('Elapsed time: %f seconds' % (toc-tic).total_seconds())

if __name__ == '__main__':
    main()
