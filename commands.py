# prepare data
cd tools/prepare_zju_mocap
python prepare_dataset.py --cfg 387.yaml
cd ../../


# Train a model by yourself. We used 4 GPUs (NVIDIA RTX 2080 Ti) to train a model
CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/human_nerf/zju_mocap/386/adventure.yaml

# For sanity check, we provide a configuration that supports training on a single GPU (NVIDIA RTX 2080 Ti). Notice the performance is not guranteed for this configuration.
CUDA_VISIBLE_DEVICES=6 python train.py --cfg configs/human_nerf/zju_mocap/387/single_gpu_test.yaml
CUDA_VISIBLE_DEVICES=7 python train.py --cfg configs/human_nerf/zju_mocap/387/single_gpu_test1.yaml

# Render the frame input (i.e., observed motion sequence).
CUDA_VISIBLE_DEVICES=1 python run.py  --type movement  --cfg configs/human_nerf/zju_mocap/377/pretrained.yaml 
CUDA_VISIBLE_DEVICES=1 python run.py  --type movement  --cfg configs/human_nerf/zju_mocap/377/adventure.yaml 
CUDA_VISIBLE_DEVICES=1 python run.py  --type movement  --cfg configs/human_nerf/zju_mocap/377/single_gpu.yaml 

# Render the motionCLIP motion
run.py 
    >> create_dataset.py
    >> func create_dataloader


# Run free-viewpoint rendering on a particular frame (e.g., frame 128).
CUDA_VISIBLE_DEVICES=1 python run.py  --type freeview --cfg configs/human_nerf/zju_mocap/377/pretrained.yaml freeview.frame_idx 128
CUDA_VISIBLE_DEVICES=1 python run.py  --type freeview --cfg configs/human_nerf/zju_mocap/377/adventure.yaml freeview.frame_idx 128
CUDA_VISIBLE_DEVICES=7 python run.py  --type freeview --cfg configs/human_nerf/zju_mocap/377/single_gpu.yaml freeview.frame_idx 128

# Render the learned canonical appearance (T-pose).
CUDA_VISIBLE_DEVICES=3 python run.py  --type tpose --cfg configs/human_nerf/zju_mocap/387/pretrained.yaml 
CUDA_VISIBLE_DEVICES=2 python run.py  --type tpose --cfg configs/human_nerf/zju_mocap/377/adventure.yaml 
CUDA_VISIBLE_DEVICES=2 python run.py  --type tpose --cfg configs/human_nerf/zju_mocap/387/single_gpu.yaml 


# Evaluation (https://github.com/escapefreeg/humannerf-eval)
# comment folder path in line 19 core/data/dataset_args.py
# uncomment folder path in line 20 core/data/dataset_args.py
cd tools/prepare_zju_mocap
python prepare_dataset_eval.py --cfg 387_eval.yaml
cd ../../
# Run eval.py to evaluate 
CUDA_VISIBLE_DEVICES=3 python eval.py --cfg configs/human_nerf/zju_mocap/387/pretrained.yaml
CUDA_VISIBLE_DEVICES=3 python eval.py --cfg configs/human_nerf/zju_mocap/387/single_gpu.yaml




CUDA_VISIBLE_DEVICES=3 python train.py --cfg configs/human_nerf/zju_mocap/377/single_gpu.yaml


CUDA_VISIBLE_DEVICES=4 python train.py --cfg configs/human_nerf/people_snapshot/male-3-casual/single_gpu.yaml
CUDA_VISIBLE_DEVICES=2 python run.py  --type tpose --cfg configs/human_nerf/people_snapshot/male-3-casual/single_gpu.yaml 

# change experiment output path
# 1. humannerf/scripts/download_model.sh  (Line 18)
# 2. humannerf/tools/prepare_zju_mocap/386.yaml (Line 2, 10)
# 3. humannerf/core/data/dataset_args.py  (Line 12, 18)
# 4. humannerf/configs/config.py  (Line 33)
# 5. cp ./377/adventure.yaml ./377/pretrained.yaml (Line 3: experiment: 'pretrained') 

# prepare people_snapshot [not yet ready]
# 1. mkdir/tools/prepare_people_snapshot
# 2. pip install PyOpenGL_accelerate==3.1.5 (required)
# 3. cp PATH_neuralbody/tools/process_snapshot.py ./tools/prepare_people_snapshot/prepare_dataset.py
# 4. cp PATH_neuralbody/tools/vis_snapshot.py ./tools/prepare_people_snapshot/vis_snapshot.py
















