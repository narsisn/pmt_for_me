
CUDA_VISIBLE_DEVICES=0
python train_net_video.py  --num-gpus 1 \
  --config-file configs/ytvis19/videomt/vit-large/videomt_segmenter_ViTL.yaml \
  --resume MODEL.WEIGHTS /mnt/adas7tb/narges/video_segmentation/pre-trained-models/dinov3_original/dinov3_repo/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  SOLVER.MAX_ITER 40000 \
  SOLVER.IMS_PER_BATCH 1 \
  OUTPUT_DIR /mnt/adas7tb/narges/frozen_videomt/ 



#!/bin/bash -l

HOSTNAME=$(hostname)

if [[ "$HOSTNAME" == *"snellius"* ]] || [[ "$HOSTNAME" == *"leonardo"* ]]; then
  echo "Activating environment..."
  eval "$(conda shell.bash hook)"
  conda activate eomt
  echo "Environment activated."
fi

EXPERIMENT_NAME="coco_pan_vit_l_next_mf_640"           # Experiment name
DEVICES=4                                              # Number of GPUs
BS=4                                                   # Per GPU batch size
DEFAULT_ROOT=/data/niccoloc/datasets/coco              # Data dir
RUN_PREFIX=""
ROOT=$DEFAULT_ROOT
SAVE_DIR=output/$EXPERIMENT_NAME
PROJECT="frozen_eomt"

case "$HOSTNAME" in
  *snellius*)
    echo "Running on Snellius"
    DATASET_NAME=$(basename "$DEFAULT_ROOT")
    ROOT=/projects/prjs1742/data/$DATASET_NAME
    RUN_PREFIX="srun "
    ;;
  *mps*)
    echo "Running on MPS"
    ROOT=/home/ncavagnero/datasets/coco
    ;;
  *leonardo*)
    echo "Running on Leonardo"
    DATASET_NAME=$(basename "$DEFAULT_ROOT")
    ROOT=/leonardo_scratch/fast/IscrC_veomt/data/$DATASET_NAME
    SAVE_DIR=/leonardo_scratch/fast/IscrC_veomt/dense_decoder_output/$EXPERIMENT_NAME
    RUN_PREFIX="srun "

    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export HF_HUB_ENABLE_HF_TRANSFER=0
    export TORCH_HOME=/leonardo_scratch/fast/IscrC_veomt/cache/torch
    export HF_HOME=/leonardo_scratch/fast/IscrC_veomt/cache/huggingface
    export WANDB_MODE=offline
    ;;
  *)
    echo "Using default configuration"
    RUN_PREFIX=""
    ;;
esac

${RUN_PREFIX}python3 main.py fit \
  -c configs/frozen/coco/panoptic/next_mf.yaml \
  --trainer.devices $DEVICES \
  --trainer.logger.init_args.save_dir $SAVE_DIR \
  --trainer.logger.init_args.name $EXPERIMENT_NAME \
  --trainer.logger.init_args.project $PROJECT \
  --data.batch_size $BS \
  --data.path $ROOT