import imp

def create_lr_updater(cfg):
    module = cfg.lr_updater_module
    lr_updater_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, lr_updater_path).update_lr
