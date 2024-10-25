# MetasurfaceVIT: A Generic Framework for Optical Inverse Design

By [Jiahao Yan](mailto:yjh20xy@gmail.com)
&nbsp;&nbsp;&nbsp;&nbsp;[Google Scholar Link](https://scholar.google.com/citations?user=LSAGvLcAAAAJ&hl=en&oi=ao)
&nbsp;&nbsp;&nbsp;&nbsp;[GitHUB Link](https://github.com/JYJiahaoYan)

## Introduction

**MetasurfaceVIT** is built for inversely design metasurfaces involving various types of amplitudes and phases engineering. This project mainly contains five parts:
1) Generate data using FDTD simulation and following calculations based on Jones Matrix.
2) Use generated data (wavelength-dependent Jones matrix) for masked pretrained.
3) Application-oriented metasurface design and Jones matrix reconstruction.
4) Fine tune model for structural parameters' generation, prediction, and evaluation.
5) Metasurface building using predicted parameters and verification based on forward network & optical simulation.

# Commit History
1) Oct 25 2024: Upload the project and ensure that small data size with basic settings are runnable
2) tbd

## Project Structure
```commandline
E:\METASURFACEVIT-MAIN
│  config.py
│  logger.py
│  lr_scheduler.py
│  main_finetune.py
│  main_metalens.py
│  main_pretrain.py
│  optimizer.py
│  readme.md
│  utils.py
│
├─data
│      data_finetune.py
│      data_recon.py
│      data_simmim.py
│      __init__.py
│
├─evaluation
│  ├─metasurface_design
│  │      image_generator.py
│  │      JM_generator.py
│  │      main.py
│  │      utils.py
│  │
│  └─metasurface_verification
│      │  main.py
│      │  matcher.py
│      │  predictor.py
│      │  visualization.py
│      │
│      └─predict_params
├─figures
│  ├─color
│  │      img10.jpg
│  │      ...
│  │
│  ├─grey
│  │      F1.jpg
│  │      ...
│  │
│  └─presentation
├─metalens_output
│      lens_construct.lsf
│      lens_simulate.lsf
│      ...
├─model
│      simmim.py
│      vision_transformer.py
│      __init__.py
│
└─preprocess
    │  data_generation.py
    │  ...
    ├─FDTD_Simulation
    │      prebuilt.fsp
    │      unit_cell.py
    │      unit_script.lsf
    │
    └─Jones_matrix_calculation
            double_cell.py
            jones_matrix.py
            jones_vector.py
            visualization.py
            __init__.py
```

## Main Results

### type1: single-wavelength multi-polarization display and hologram

**tbd**

### type2: multi-wavelength multi-polarization display and hologram

**tbd**

### type3: RGB three channels display and hologram

**tbd**

### type4: broadband (covering visible wavelengths) metalens

**tbd**


## Getting Started

### Installation

- Install `CUDA 11.6` with `cuDNN 8` following the official installation guide of [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive).

- Setup environment:
```bash
conda create -n MetasurfaceVIT python=3.8
conda activate MetasurfaceVIT
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
conda install matplotlib
conda install pillow
conda install numpy
pip install timm
pip install termcolor
pip install scipy
pip install yacs
## Nvidia apex is optional. Our code also considers the implementation of pytorch amp.
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir ./
```

### Generate and calculate optical data
```bash
# step1: generate pretrained dataset
# I highly recommend to use small data first to go through the whole workflow
python preprocess/data_generation.py --min_size 40 --max_size 200 --step 20 --points 10 --visualize true
# then you can use default value to generate ~20M data, which is a suitable size to perform pretrain
python preprocess/data_generation.py

# step2: generate finetune dataset
# small data size
python preprocess/data_generation.py --min_size 40 --max_size 200 --step 20 --points 10 --visualize true --finetune --finetune_factor 1
# large data size (tbd)

```

### Masked pre-training
```bash
# for small datasize, you could run following command on single GPU
python main_pretrain.py --epoch 10 --mask_type 0 --data_size 1 --data_start 2
# or use
python main_pretrain.py --epoch 10 --mask_type 0 --data_size 1 --data_start 2 --amp_type pytorch

# for large datasize (~20M data), use distributed training command (tbd):
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_pretrain.py
```

### Metasurface Design
```bash
# please check this python file and explore different arguments
python evaluation/metasurface_design/main.py --pretrain_path preprocess/training_data_2 --design_type 1 --visualize
python evaluation/metasurface_design/main.py --pretrain_path preprocess/training_data_2 --design_type 2 --visualize
python evaluation/metasurface_design/main.py --pretrain_path preprocess/training_data_2 --design_type 3 --visualize
python evaluation/metasurface_design/main.py --pretrain_path preprocess/training_data_2 --design_type 4 --visualize --amplitude all
```

### Jones Matrix reconstruction
```bash
python main_pretrain.py --recon --recon_type 1
python main_pretrain.py --recon --recon_type 2
python main_pretrain.py --recon --recon_type 3
# for type 4 (metalens), in some case (when self.amplitude == 'all' in evaluation/metasurface_design/JM_generator.py),
# there is no need to do reconstruction. but to be consistent, you can still follow this workflow.
python main_pretrain.py --recon --recon_type 4
```

### Fine-tuning pre-trained models & parameter prediction
```bash
# step 1: finetune
# simple version:
python main_finetune.py --epoch 100 --data_folder_name finetune_data_1 
# you might wanna use distributed training
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> main_finetune.py

# step2: parameter prediction 
# for design_type 1-3
python main_finetune.py --eval --data_folder_name finetune_data_1 --recon_type 1 --treatment 2024-10-14
python main_finetune.py --eval --data_folder_name finetune_data_1 --recon_type 2 --treatment 2024-10-14
python main_finetune.py --eval --data_folder_name finetune_data_1 --recon_type 3 --treatment 2024-10-14
# for design_type 4 (iterated process)
python main_metalens.py --eval --data_folder_name finetune_data_1
```

### Metasurface forward prediction & metalens simulation
```bash
# for design_type 1-3
# verify several configurations: 
# add --train if there is no suitable trained model for your design type.
# without --train, this code will automatically find the matched model structure with the latest time stamp and evaluate.
python evaluation/metasurface_verification/main.py --verify_type predictor --network MLP <--train> --design_type 1 --treatment 2024-10-14 --finetune_folder finetune_data_1
python evaluation/metasurface_verification/main.py --verify_type predictor --network CNN <--train> --design_type 1 --treatment 2024-10-14 --finetune_folder finetune_data_1
python evaluation/metasurface_verification/main.py --verify_type predictor --network MLP <--train> --design_type 2 --treatment 2024-10-14 --finetune_folder finetune_data_1
python evaluation/metasurface_verification/main.py --verify_type predictor --network CNN <--train> --design_type 2 --treatment 2024-10-14 --finetune_folder finetune_data_1
python evaluation/metasurface_verification/main.py --verify_type predictor --network MLP <--train> --design_type 3 --treatment 2024-10-14 --finetune_folder finetune_data_1
python evaluation/metasurface_verification/main.py --verify_type predictor --network CNN <--train> --design_type 3 --treatment 2024-10-14 --finetune_folder finetune_data_1
# you can also use Matcher to perform forward prediction
python evaluation/metasurface_verification/main.py --verify_type matcher --design_type 1 --treatment 2024-10-14 --finetune_folder finetune_data_1
# for design_type 4, please navigate to metalens_output folder and interact with FDTD files
```
