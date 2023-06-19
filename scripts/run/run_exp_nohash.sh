CUDA_VISIBLE_DEVICES=5 python train.py --cfg configs/human_nerf/zju_mocap/387/single_gpu_nohash.yaml
CUDA_VISIBLE_DEVICES=5 python eval.py --cfg configs/human_nerf/zju_mocap/387/single_gpu_nohash.yaml