import imp

def _query_trainer(cfg):
    module = cfg.trainer_module
    trainer_path = module.replace(".", "/") + ".py"
    trainer = imp.load_source(module, trainer_path).Trainer
    print('use trainer: ' + trainer_path) # dj
    return trainer

def create_trainer(cfg, network, optimizer):
    Trainer = _query_trainer(cfg)
    return Trainer(cfg, network, optimizer)
