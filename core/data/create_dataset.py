import os
import imp
import torch

from core.utils.file_util import list_files
from .dataset_args import DatasetArgs

def _query_dataset(cfg, data_type):
    module = cfg[data_type].dataset_module
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).Dataset

def _get_total_train_imgs(dataset_path):
    train_img_paths = list_files(os.path.join(dataset_path, 'images'), exts=['.png'])
    return len(train_img_paths)

def create_dataset(cfg, data_type='train'):
    dataset_name = cfg[data_type].dataset
    args = DatasetArgs(cfg).get(dataset_name)

    # customize dataset arguments according to dataset type
    args['bgcolor'] = None if data_type == 'train' else cfg.bgcolor
    
    total_train_imgs = _get_total_train_imgs(args['dataset_path'])

    if data_type == 'progress' or data_type == 'tpose' or data_type == 'freeview':
        args['skip'] = total_train_imgs // 16
        args['maxframes'] = 16

    dataset = _query_dataset(cfg, data_type)
    dataset = dataset(cfg, **args)
    return dataset

def create_dataloader(cfg, data_type='train'):
    cfg_node = cfg[data_type]
    batch_size = cfg_node.batch_size
    shuffle = cfg_node.shuffle
    drop_last = cfg_node.drop_last

    dataset = create_dataset(cfg, data_type=data_type)
    
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
    )
    return data_loader
