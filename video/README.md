# PMT: Plain Mask Transformer — Video Segmentation
**CVPR 2026 Workshop** · [📄 Paper](https://arxiv.org/abs/2603.25398)

**[Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ), [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ), [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ), [Daan de Geus](https://ddegeus.github.io)**

Eindhoven University of Technology

## Overview

This directory contains the **video segmentation** component of [PMT (Plain Mask Transformer)](../README.md).

PMT is a fast Transformer-based segmentation model that operates on top of **frozen** Vision Foundation Model (VFM) features. The key idea is the **Plain Mask Decoder (PMD)**: a lightweight Transformer decoder that processes queries and frozen patch tokens jointly — without finetuning the encoder — keeping it shareable across tasks.

This video extension brings PMT to online video instance, panoptic, and semantic segmentation. Temporal reasoning is handled inside the decoder via a compact propagation mechanism, without modifying the frozen ViT encoder.



## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n pmt python==3.13.2
conda activate pmt
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'  
pip install git+https://github.com/cocodataset/panopticapi.git
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation

[Download and prepare the datasets.](datasets/README.md)  


## Usage

### Evaluation

To evaluate a pre-trained PMT model, first prepare the datasets by following the instructions in this [link](datasets/README.md) and download the trained weights from [DINOv2 models](model_zoo/dinov2.md) or [DINOv3 models](model_zoo/dinov3.md). Once these are set up, run:

```bash
python train_net_video.py \
  --num-gpus 1 \
  --config-file /path/to/config.yaml \
  --eval-only MODEL.WEIGHTS /path/to/weight.pth \
  MODEL.MODEL.BACKBONE.TEST.WINDOW_SIZE 1 \ 
  OUTPUT_DIR /path/to/output
```

🔧 Replace `/path/to/config.yaml` with the path to the config file.  
🔧 Replace `/path/to/weight.pth` with the path to the checkpoint to evaluate.   
🔧 Replace `/path/to/output` with the path to the output folder.  
🔧 Change the value of `--num-gpus` to the number of GPUs available to you.

For detailed instructions on running evaluation on different datasets, see [Evaluation](model_zoo/evaluation.md).

### Benchmark

To calculate the FPS and GFLOPs, run: 

```bash
python benchmark.py \
  --task fps \
  --config-file    /path/to/config.yaml \
  --model-weights  /path/to/weight.pth  \
  --warmup-iters 100 

export TIMM_FUSED_ATTN=0 
python benchmark.py \
  --task flops \
  --config-file    /path/to/config.yaml \
  --model-weights  /path/to/weight.pth  
```

🔧 Replace `/path/to/config.yaml` with the path to the config file.  
🔧 Replace `/path/to/weight.pth` with the path to the checkpoint to evaluate.   

## Demo

We provide example visualizations below.  


<img src="./docs/videos/1f17cd7c_5.gif" width="400"/> <img src="./docs/videos/d4f4cf55_5.gif" width="400" /> 
<img src="./docs/videos/1010_kI0mOZirPGs_5.gif" width="400" /> <img src="./docs/videos/1975_1qyIMfzlXnY_5.gif" width="400" />
<!-- <img src="./docs/videos/35d5e5149d_5.gif" width="400" />  -->



To generate additional visualization samples, please use the code in [Visualization](model_zoo/visualization.md).



## Model Zoo

We provide pre-trained weights for both DINOv2- and DINOv3-based PMT models.

- **[DINOv2 Models](model_zoo/dinov2.md)** - Original published results and pre-trained weights.
- **[DINOv3 Models](model_zoo/dinov3.md)** - DINOv3-based models and pre-trained weights.

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@article{Cavagnero2026PMT,
  author     = {Cavagnero, Niccol\`{o} and Norouzi, Narges and Dubbelman, Gijs and {de Geus}, Daan},
  title      = {{PMT: Plain Mask Transformer for Image and Video Segmentation with Frozen Vision Encoders}},
  journal    = {arxiv},
  year       = {2026},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:
- [EoMT](https://github.com/tue-mps/eomt) (MIT License)  
- [VidEoMT](https://github.com/tue-mps/videomt) (MIT License)  
- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [CAVIS](https://github.com/Seung-Hun-Lee/CAVIS) (MIT License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)

