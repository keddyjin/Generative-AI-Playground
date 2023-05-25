- [1 Install threestudio](#1-install-threestudio)
- [2 BUGs](#2-bugs)
  - [2.1](#21)
  - [2.2](#22)
  - [2.3](#23)
  - [2.4](#24)
  - [2.5](#25)
  - [2.6](#26)
  - [2.7](#27)
- [3 Running](#3-running)
  - [Running Magic3D](#running-magic3d)
    - [Coarse stage](#coarse-stage)
    - [Refine stage](#refine-stage)


# 1 Install threestudio
Install PyTorch >= 1.12.  
```
# update CUDA version to 11.8
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

# create conda env and install pytorch
conda create --name threestudio python
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```
Install ninja to speed up the compilation of CUDA extensions:  
`pip install ninja`  
Install dependencies:  
```
export PATH=/usr/local/cuda/bin:$PATH
pip install -r requirements.txt
```
The best-performing models in threestudio uses the newly-released T2I model [DeepFloyd IF](https://github.com/deep-floyd/IF) which currently requires signing a license agreement. If you would like use these models, you need to accept the license on the [model card of DeepFloyd IF](https://huggingface.co/DeepFloyd/IF-I-XL-v1.0), and login in the Hugging Face hub in terminal by `huggingface-cli login`.  
```
pip install huggingface_hub
# You already have it if you installed transformers or datasets

huggingface-cli login
#Log in using a token from huggingface.co/settings/tokens
```

# 2 BUGs
## 2.1
```
╭──────────────────────────────────────────────────────────────────────────────────────────────────╮
│ B ... Nc                                                                                         │
│   ▲                                                                                              │
╰──────────────────────────────────────────────────────────────────────────────────────────────────╯
SyntaxError: invalid syntax  
```
Solution: update to the newest codebase. https://github.com/threestudio-project/threestudio/issues/62  
`git pull`

## 2.2
```
/usr/include/c++/11/bits/std_function.h:530:146: error: parameter packs not expanded with ‘...’:
        530 |         operator=(_Functor&& __f)
            |                                                                                                                                                  ^
      /usr/include/c++/11/bits/std_function.h:530:146: note:         ‘_ArgTypes’
      error: command '/usr/bin/nvcc' failed with exit code 1
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for nerfacc

libcusparse.so.11: cannot open shared object file: No such file or directory  
RuntimeError:
        CUDA Setup failed despite GPU being available. Please run the following command to get more information:

        python -m bitsandbytes

        Inspect the output of the command and see if you can locate CUDA libraries. You might need to add them
        to your LD_LIBRARY_PATH. If you suspect a bug, please take the information from python -m bitsandbytes
        and open an issue at: https://github.com/TimDettmers/bitsandbytes/issues
```
Solution: change CUDA and Pytorch environment
```
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh cuda_11.8.0_520.61.05_linux.run

conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# add PATH and LD_LIBRARY_PATH in .bashrc file
```

## 2.3
```
AttributeError: partially initialized module 'cv2' has no attribute 'gapi_wip_gst_GStreamerPipeline' (most likely due to a circular
import)
or
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```
Solution: 
```
pip uninstall opencv-python
pip uninstall opencv-contrib-python
pip uninstall opencv-python-headless
pip uninstall opencv-contrib-python-headless
pip install opencv-python==4.5.5.64
pip install opencv-python-headless==4.5.5.64

```

## 2.4
```
A matching Triton is not available, some optimizations will not be enabled.
Error caught was: No module named 'triton'
```
Solution: `pip install triton`

## 2.5
```
File "/workspace/host_data/public/dataset/keddyj/threestudio/threestudio/models/prompt_processors/deepfloyd_prompt_processor.py", line 80, in spawn_func
    text_embeddings = text_encoder(
RuntimeError: expected scalar type Float but found Half        
```
Solution: add `with torch.autocast("cuda"):` before `text_embeddings = text_encoder(` in line 80 of deepfloyd_prompt_processor.py, ref: https://huggingface.co/CompVis/stable-diffusion-v1-4/discussions/10

## 2.6
When running `apt update`, there's error
```
At least one invalid signature was encountered.
```
Solution: running with `apt update --allow-unauthenticated --allow-insecure-repositories`

## 2.7
```
File "/home/nvidia/anaconda3/envs/threestudio/lib/python3.11/site-packages/transformers/models/clip/tokenization_clip.py", line 322, in __init__
    with open(vocab_file, encoding="utf-8") as vocab_handle:
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
TypeError: expected str, bytes or os.PathLike object, not NoneType
```
Solution: It's caused by network connection

# 3 Running 
## Running Magic3D
### Coarse stage
First train the coarse stage NeRF:
```
python launch.py --config configs/magic3d-coarse-if.yaml --train --gpu 0 system.prompt_processor.prompt="a blue poison-dart frog sitting on a water lily"
```
The coarse stage result:
```
[INFO] ModelCheckpoint(save_last=True, save_top_k=-1, monitor=None) will duplicate the last checkpoint saved.
[INFO] Using 16bit Automatic Mixed Precision (AMP)
[INFO] GPU available: True (cuda), used: True
[INFO] TPU available: False, using: 0 TPU cores
[INFO] IPU available: False, using: 0 IPUs
[INFO] HPU available: False, using: 0 HPUs
[INFO] You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
[INFO] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[INFO]
  | Name       | Type                           | Params
--------------------------------------------------------------
0 | material   | DiffuseWithPointLightMaterial  | 0
1 | background | NeuralEnvironmentMapBackground | 448
2 | geometry   | ImplicitVolume                 | 12.6 M
3 | renderer   | NeRFVolumeRenderer             | 0
--------------------------------------------------------------
12.6 M    Trainable params
0         Non-trainable params
12.6 M    Total params
50.419    Total estimated model params size (MB)
[INFO] Validation results will be saved to outputs/magic3d-coarse-if/a_blue_poison-dart_frog_sitting_on_a_water_lily@20230524-232812/save
[INFO] Using prompt [a blue poison-dart frog sitting on a water lily] and negative prompt []

Epoch 0: : 10000it [25:24,  6.56it/s][INFO] `Trainer.fit` stopped: `max_steps=10000` reached.
Epoch 0: : 10000it [25:24,  6.56it/s]
[INFO] You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
[INFO] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/nvidia/anaconda3/envs/threestudio/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:430: PossibleUserWarning: The dataloader, test_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 24 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 120/120 [00:25<00:00,  4.68it/s]
[INFO] Test results saved to outputs/magic3d-coarse-if/a_blue_poison-dart_frog_sitting_on_a_water_lily@20230524-232812/save
```
Coarse result |
:-: |
<video src='https://github.com/keddyjin/Generative-AI-Playground/assets/5978120/6936aafd-1031-4a8c-9977-27c6b9ff1872' width=180/> |

### Refine stage
Then convert the NeRF from the coarse stage to DMTet and train with differentiable rasterization:
```
python launch.py --config configs/magic3d-refine-sd.yaml --train --gpu 0 system.prompt_processor.prompt="a blue poison-dart frog sitting on a water lily" system.from_coarse=outputs/magic3d-coarse-if/a_blue_poison-dart_frog_sitting_on_a_water_lily@20230524-232812/ckpts/last.ckpt
```
The refine stage result:
```
[INFO] Initializing from coarse stage ...
[INFO] Automatically determined isosurface threshold: 43.917442321777344
[INFO] Automatically determined isosurface threshold: 56.31735610961914
[INFO] ModelCheckpoint(save_last=True, save_top_k=-1, monitor=None) will duplicate the last checkpoint saved.
[INFO] Using 16bit Automatic Mixed Precision (AMP)
[INFO] GPU available: True (cuda), used: True
[INFO] TPU available: False, using: 0 TPU cores
[INFO] IPU available: False, using: 0 IPUs
[INFO] HPU available: False, using: 0 HPUs
[INFO] You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
[INFO] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
[INFO]
  | Name       | Type                           | Params
--------------------------------------------------------------
0 | material   | DiffuseWithPointLightMaterial  | 0
1 | background | NeuralEnvironmentMapBackground | 448
2 | geometry   | TetrahedraSDFGrid              | 13.7 M
3 | renderer   | NVDiffRasterizer               | 0
--------------------------------------------------------------
13.7 M    Trainable params
448       Non-trainable params
13.7 M    Total params
54.849    Total estimated model params size (MB)
[INFO] Validation results will be saved to outputs/magic3d-refine-sd/a_blue_poison-dart_frog_sitting_on_a_water_lily@20230525-001155/save
[INFO] Using prompt [a blue poison-dart frog sitting on a water lily] and negative prompt []

Epoch 0: : 5000it [12:14,  6.81it/s][INFO] `Trainer.fit` stopped: `max_steps=5000` reached.
Epoch 0: : 5000it [12:14,  6.81it/s]
[INFO] You are using a CUDA device ('NVIDIA RTX A6000') that has Tensor Cores. To properly utilize them, you should set `torch.set_float32_matmul_precision('medium' | 'high')` which will trade-off precision for performance. For more details, read https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html#torch.set_float32_matmul_precision
[INFO] LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
/home/nvidia/anaconda3/envs/threestudio/lib/python3.11/site-packages/pytorch_lightning/trainer/connectors/data_connector.py:430: PossibleUserWarning: The dataloader, test_dataloader, does not have many workers which may be a bottleneck. Consider increasing the value of the `num_workers` argument` (try 24 which is the number of cpus on this machine) in the `DataLoader` init to improve performance.
  rank_zero_warn(
Testing DataLoader 0: 100%|██████████████████████████████████████████████████████████| 120/120 [00:03<00:00, 31.81it/s]
[INFO] Test results saved to outputs/magic3d-refine-sd/a_blue_poison-dart_frog_sitting_on_a_water_lily@20230525-101624/save
```
Refine result |
:-: |
<video src='https://github.com/keddyjin/Generative-AI-Playground/assets/5978120/a2f5e084-8224-4bfc-988e-2797e81ac81f' width=180/> |
