class DatasetArgs(object):
    def __init__(self, cfg):
        self.dataset_attrs = {}

        subjects = ['313', '315', '377', '386', '387', '390', '392', '393', '394']

        if cfg.category == 'superhumannerf' and cfg.task == 'zju_mocap': # dj
            for sub in subjects:
                self.dataset_attrs.update({
                    f"zju_{sub}_train": {
                        # "dataset_path": f"dataset/zju_mocap/{sub}",
                        "dataset_path": f"../datasets/zju_mocap/{sub}", # dj
                        "keyfilter": cfg.train_keyfilter,
                        "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    },
                    f"zju_{sub}_test": {
                        # "dataset_path": f"dataset/zju_mocap/{sub}",
                        #"dataset_path": f"../datasets/zju_mocap/{sub}", # dj
                        "dataset_path": f"../datasets/zju_mocap/{sub}_eval", # dj for evaluation/progress
                        "keyfilter": cfg.test_keyfilter,
                        "ray_shoot_mode": 'image',
                        "src_type": 'zju_mocap'
                    },
                })


        subjectsPS = ['male-3-casual']
        if cfg.category == 'superhumannerf' and cfg.task == 'people_snapshot': # dj
            for sub in subjectsPS:
                self.dataset_attrs.update({
                    f"{sub}_train": {
                        "dataset_path": f"../datasets/people_snapshot/{sub}", # dj
                        "keyfilter": cfg.train_keyfilter,
                        "ray_shoot_mode": cfg.train.ray_shoot_mode,
                    },
                    f"{sub}_test": {
                        "dataset_path": f"../datasets/people_snapshot/{sub}", # dj
                        "keyfilter": cfg.test_keyfilter,
                        "ray_shoot_mode": 'image',
                        "src_type": 'people_snapshot'
                    },
                })

        if cfg.category == 'superhumannerf' and cfg.task == 'wild':
            self.dataset_attrs.update({
                "wild_pitching_train": {
                    "dataset_path": '../datasets/wild/pitching',
                    "keyfilter": cfg.train_keyfilter,
                    "ray_shoot_mode": cfg.train.ray_shoot_mode,
                },
                "wild_pitching_test": {
                    "dataset_path": '../datasets/wild/pitching',  
                    "keyfilter": cfg.test_keyfilter,
                    "ray_shoot_mode": 'image',
                    "src_type": 'wild'
                },
            })

    def get(self, name):
        attrs = self.dataset_attrs[name]
        return attrs.copy()
