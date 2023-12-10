# Superhumannerf

## Prerequisite

### Create environment

Create and activate a virtual environment.

    conda create --name superhumannerf python=3.8
    conda activate superhumannerf

Install pytorch with cuda11.7

    pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117

Install the required packages.

    pip install -r requirements.txt

Make sure cuda version is right

    export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-11.7/bin:$PATH

Install tinycudann

    pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch

Install torch-scatter

    pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cu117.html


### Download SMPL model

Download the gender neutral SMPL model from [here](https://smplify.is.tue.mpg.de/), and unpack **mpips_smplify_public_v2.zip**.

Copy the smpl model.

    SMPL_DIR=/path/to/smpl
    MODEL_DIR=$SMPL_DIR/smplify_public/code/models
    cp $MODEL_DIR/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl third_parties/smpl/models

Follow [this page](https://github.com/vchoutas/smplx/tree/master/tools) to remove Chumpy objects from the SMPL model.

## Run on ZJU-Mocap Dataset

### Prepare a dataset

First, download ZJU-Mocap dataset from [here](https://github.com/zju3dv/neuralbody/blob/master/INSTALL.md#zju-mocap-dataset). 

Second, modify the yaml file of subject 387 at `tools/prepare_zju_mocap/387.yaml`. In particular,  `zju_mocap_path` should be the directory path of the ZJU-Mocap dataset.

```
dataset:
    zju_mocap_path: /path/to/zju_mocap
    subject: '387'
    sex: 'neutral'

...
```
    
Finally, run the data preprocessing script.

    python ./tools/prepare_dataset.py

### `Train models`

    python train.py


### `Render output`

Render the frame input (i.e., observed motion sequence).

    python run.py \
        --type movement \
        --cfg configs/human_nerf/zju_mocap/387/adventure.yaml 

Run free-viewpoint rendering on a particular frame (e.g., frame 128).

    python run.py \
        --type freeview \
        --cfg configs/human_nerf/zju_mocap/387/adventure.yaml \
        freeview.frame_idx 128


Render the learned canonical appearance (T-pose).

    python run.py \
        --type tpose \
        --cfg configs/human_nerf/zju_mocap/387/adventure.yaml 

## Acknowledgement

The implementation is modified from [HumanNeRF](https://github.com/chungyiweng/humannerf), which took reference from [NeRF-PyTorch](https://github.com/yenchenlin/nerf-pytorch), [Neural Body](https://github.com/zju3dv/neuralbody), [Neural Volume](https://github.com/facebookresearch/neuralvolumes), [LPIPS](https://github.com/richzhang/PerceptualSimilarity), and [YACS](https://github.com/rbgirshick/yacs). Thanks for their great work.