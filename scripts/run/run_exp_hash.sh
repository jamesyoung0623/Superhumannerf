CUDA_VISIBLE_DEVICES=7 python train.py --cfg configs/human_nerf/zju_mocap/387/single_gpu_hash.yaml
CUDA_VISIBLE_DEVICES=7 python eval.py --cfg configs/human_nerf/zju_mocap/387/single_gpu_hash.yaml