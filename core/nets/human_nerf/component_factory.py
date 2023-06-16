import imp

def load_positional_embedder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    # print(module_path) (core/nets/human_nerf/embedders/hannw_fourier.py)
    return imp.load_source(module, module_path).get_embedder

def load_canonical_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    return imp.load_source(module, module_path).CanonicalMLP

def load_mweight_vol_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    # print(module_path) (core/nets/human_nerf/mweight_vol_decoders/deconv_vol_decoder.py)
    return imp.load_source(module, module_path).MotionWeightVolumeDecoder

def load_pose_decoder(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    # print(module_path) (core/nets/human_nerf/pose_decoders/mlp_delta_body_pose.py)
    return imp.load_source(module, module_path).BodyPoseRefiner

def load_non_rigid_motion_mlp(module_name):
    module = module_name
    module_path = module.replace(".", "/") + ".py"
    # print(module_path) (core/nets/human_nerf/non_rigid_motion_mlps/mlp_offset.py)
    return imp.load_source(module, module_path).NonRigidMotionMLP
